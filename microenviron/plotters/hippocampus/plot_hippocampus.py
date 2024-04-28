#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : generate_me_map.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-09
#   Description  : 
#
#================================================================
import os
import sys
import random
import numpy as np
import numbers
import pickle
import pandas as pd
from skimage import exposure, filters, measure
from skimage import morphology
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
import matplotlib
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from fil_finder import FilFinder2D
import astropy.units as u
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

from image_utils import get_mip_image, image_histeq, get_longest_skeleton
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, ANATOMY_TREE_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, \
                                get_brain_mask2d, get_section_boundary, detect_edges2d
from anatomy.anatomy_core import parse_ana_tree
from image_utils import crop_nonzero_mask

sys.path.append('../..')
from shape_normalize import stretching2d, map_to_longitudinal_space
from generate_me_map import process_mip, generate_me_maps, colorize_atlas2d_cv2

# features selected by mRMR
__MAP_FEATS__ = ('Length', 'AverageFragmentation', 'AverageContraction')


def scatter_hist(x, y, colors, ax, ax_histx, ax_histy, xylims):

    def get_values(xy, cs, bins):
        bw = (xy.max() - xy.min()) / bins
        data = []
        xys = []
        for xyi in np.arange(xy.min(), xy.max(), bw):
            xym = (xy >= xyi) & (xy < xyi+bw)
            if xym.sum() < 2: continue
            vs = cs[xym].mean()
            xys.append(xyi + bw/2)
            data.append(vs)
        return xys, data
            

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    xmin, xmax = x.min(), x.max()
    ax.scatter(x, y, c=colors, s=9)
    ax.set_xlim(*(xylims[0]))
    ax.set_ylim(*(xylims[1]))
    ax.set_xlabel('Longitudinal distance (mm)')

    nbins = 15
    ax_histx.plot(*get_values(x, colors[:,0], nbins), c='red')
    ax_histx.plot(*get_values(x, colors[:,1], nbins), c='green')
    ax_histx.plot(*get_values(x, colors[:,2], nbins), c='blue')
    ax_histy.plot(*get_values(y, colors[:,0], nbins)[::-1], c='red')
    ax_histy.plot(*get_values(y, colors[:,1], nbins)[::-1], c='green')
    ax_histy.plot(*get_values(y, colors[:,2], nbins)[::-1], c='blue')
    
    ax_histx.set_ylabel('Feature')
    ax_histx.yaxis.set_label_position('right') 
    ax_histx.set_ylim(0, 1)
    ax_histx.set_yticks([0,1])
    
    ax_histy.set_xlabel('Feature')
    ax_histy.set_xlim(0, 1)
    ax_histy.set_xticks([0,1])


def plot_region_feature_sections(mefile, rname='MOB', r316=False, flipLR=True, thickX2=10, debug=True):
    df = pd.read_csv(mefile, comment='#', index_col=0)
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    if r316:
        rkey = 'region_name_r316'
        mask = load_image(MASK_CCF25_R314_FILE)
    else:
        rkey = 'region_name_r671'
        mask = load_image(MASK_CCF25_FILE)
    ana_tree = parse_ana_tree(keyname='name')

    if type(rname) is list:
        sel_mask = df[rkey].isin(rname)
        rmask = np.zeros_like(mask)
        for ri in rname:
            idx = ana_tree[ri]['id']
            rmask = rmask | (mask == idx)
            
        out_prefix = 'tmp'
    else:
        sel_mask = df[rkey] == rname
        idx = ana_tree[rname]['id']
        rmask = mask == idx
        out_prefix = rname

    dfr = df[keys][sel_mask]
    coords = df[['soma_x', 'soma_y', 'soma_z']][sel_mask].values / 1000
    if flipLR:
        zdim = 456
        zcoord = zdim * 25. / 1000
        right = np.nonzero(coords[:,2] > zcoord/2)[0]
        coords[right, 2] = zcoord - coords[right, 2]
        rmask[zdim//2:] = 0

    if 1:
        # keep the orginal multiple regions
        mm = mask.copy()
        mm[rmask == 0] = 0
        rmask = mm

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0]
    dfc[dfc > 255] = 255

    # get the boundary of region
    sub_mask, (zmin, zmax, ymin, ymax, xmin, xmax) = crop_nonzero_mask(rmask, pad=0)
    memap = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)
    
    coords_s = np.floor(coords * 40).astype(int)
    memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = dfc.values

    mips = []
    shape3d = mask.shape
    axid = 2
    cmap = {
        382: ('brown', 'CA1'),
        423: ('olive', 'CA2'),
        463: ('crimson', 'CA3'),
        502: ('navy', 'SUB'),
        484682470: ('purple', 'ProS'),
        10703: ('green', 'DG-mo'),
        10704: ('orange', 'DG-po'),
        632: ('blue', 'DG-sg'),
    }
    lims = {
        0: ((0,2.6), (-1,1)),
        1: ((0,4.2), (-1,1.2)),
        2: ((0,4.8), (-1,1.6)),
        3: ((0,12.4), (-1, 1.2)),
        4: ((0,11), (-1.5, 1.5)),
        5: ((0,8), (-1.5, 1.6)),
        6: ((0,6), (-1.5, 1))
    }

    for isid, sid in enumerate(range(0, xmax-xmin-thickX2-1, thickX2*2)):
        sid = sid + thickX2
        cur_memap = memap.copy()
        cur_memap[:,:,:sid-thickX2] = 0
        cur_memap[:,:,sid+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())

        mip = get_mip_image(cur_memap, axid)
        
        figname = f'{out_prefix}_section{sid:03d}.png'
        process_mip(mip, sub_mask, axis=axid, figname=figname, sectionX=sid, with_outline=False, pt_scale=5, b_scale=0.5)

        section_mask = sub_mask[:,:,sid]
        
        
        # principal axis for each section
        main_axis, pcoords = get_longest_skeleton(section_mask, is_3D=False, smoothing=True)
        # We would like to unify the definition of left right
        anchor_pt = np.array([main_axis.shape[0], 0])
        anchor2p1 = np.linalg.norm(pcoords[0] - anchor_pt)
        anchor2p2 = np.linalg.norm(pcoords[-1] - anchor_pt)
        if anchor2p1 > anchor2p2:
            pcoords = pcoords[::-1]

        if debug:
            mm = np.hstack(((section_mask > 0).astype(np.uint8), main_axis)) * 255
            cv2.imwrite(f'temp_{figname}', mm)
        # map the points to longitudinal space
        edges = detect_edges2d(section_mask > 0)
        ecoords = np.array(edges.nonzero()).transpose()
        #stretching2d(pcoords, ecoords, ecoords, visualize=True)
        
        # get the coordinates and colors of the current section
        neuron_mask = cur_memap.sum(axis=-1) > 0
        coords = np.array(neuron_mask.nonzero()[:2]).transpose()
        me_colors = cur_memap[neuron_mask] / 255.
        lcoords = map_to_longitudinal_space(section_mask, pcoords, coords)
        

        if 1:
            # also save the mask
            img_mask = np.zeros((*section_mask.shape, 4), dtype=np.uint8)
            for rid, cname in cmap.items():
                rgba = [int(c*255) for c in colors.to_rgba(cname[0])]
                #import ipdb; ipdb.set_trace()
                rgba[:3] = rgba[:3][::-1]
                img_mask[section_mask == rid] = rgba
            # we would like to show the longitudinal path for each section
            img_mask[main_axis] = (0,255,255,255)   # yellow

            if axid != 0:
                img_mask = cv2.rotate(img_mask, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(f'mask_{figname}', img_mask)

            # generate the color legend for mask
            xa = np.random.random((5))
            xb = np.random.random((5))
            for rid, rcolor in cmap.items():
                plt.scatter(xa, xb, c=rcolor[0], label=rcolor[-1])
            plt.legend(frameon=False, ncol=4, handletextpad=0.2)
            plt.savefig('color_legend.png', dpi=300)
            plt.close()


        
        if debug:
            # Create a Figure, which doesn't have to be square.
            sns.set_theme(style="ticks", font_scale=1.4)
            fig = plt.figure(layout='constrained')
            # Create the main axes, leaving 25% of the figure space at the top and on the
            # right to position marginals.
            ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
            # The main axes' aspect can be fixed.
            ax.set(aspect=1)
            # Create marginal axes, which have 25% of the size of the main axes.  Note that
            # the inset axes are positioned *outside* (on the right and the top) of the
            # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
            # less than 0 would likewise specify positions on the left and the bottom of
            # the main axes.
            ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
            ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
            # Draw the scatter plot and marginals.
            # transform the coordinates from pixel space to physical space
            lcoords *= 0.04 # in mm

            scatter_hist(lcoords[:,1], lcoords[:,0], me_colors, ax, ax_histx, ax_histy, lims[isid])
            plt.savefig(f'{out_prefix}_section{sid:03d}_stretched.png', dpi=300)
            plt.close()
            

        # load and remove the zero-alpha block
        img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
        wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
        ws, we = wnz[0], wnz[-1]
        hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
        hs, he = hnz[0], hnz[-1]
        img = img[hs:he+1, ws:we+1]
        # set the alpha of non-brain region as 0
        img[img[:,:,-1] == 1] = 0
        if axid != 0:   # rotate 90
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            #
        cv2.imwrite(figname, img)
 


    

if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    mapfile = 'microenviron_map'
    scale = 25.
    flip_to_left = True
    mode = 'composite'
    axids = (2,)
    findex = 0
    fmt = 'png'

    if 0:
        generate_me_maps(mefile, outfile=mapfile, flip_to_left=flip_to_left, mode=mode, findex=findex, fmt=fmt, axids=axids)

    if 0:
        for sectionX in range(20, 528, 40):
            colorize_atlas2d_cv2(annot=True, fmt=fmt, sectionX=sectionX)

    if 1:
        mefile = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
        swcdir = '/PBshare/SEU-ALLEN/Users/Sujun/230k_organized_folder/cropped_100um/'

        #rname = ['ACAv2/3', 'AIv2/3', 'GU2/3', 'MOp2/3', 'MOs2/3', 'ORBl2/3', 'ORBm2/3', 'ORBvl2/3', 'PL2/3', 'RSPv2/3', 'SSp-m2/3', 'SSp-n2/3']
        rname = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
        plot_region_feature_sections(mefile, rname)
   
