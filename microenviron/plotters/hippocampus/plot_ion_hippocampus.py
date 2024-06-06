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
from generate_me_map import generate_me_maps, colorize_atlas2d_cv2

# features selected by mRMR
#__MAP_FEATS__ = ('Length', 'AverageFragmentation', 'AverageContraction')
from config import mRMR_f3 as __MAP_FEATS__


def process_features(mefile, scale=25.):
    df = pd.read_csv(mefile, index_col=0)

    feat_names = [fn for fn in __MAP_FEATS__]
    # scaling the coordinates to CCFv3-25um space
    df['soma_x'] /= scale
    df['soma_y'] /= scale
    df['soma_z'] /= scale
    # we should remove the out-of-region coordinates
    zdim,ydim,xdim = (456,320,528)   # dimensions for CCFv3-25um atlas
    in_region = (df['soma_x'] >= 0) & (df['soma_x'] < xdim) & \
                (df['soma_y'] >= 0) & (df['soma_y'] < ydim) & \
                (df['soma_z'] >= 0) & (df['soma_z'] < zdim)
    df = df[in_region]
    print(f'Filtered out {in_region.shape[0] - df.shape[0]}')

    return df, feat_names

def plot_section_outline(mask, axis=0, sectionX=None, ax=None, with_outline=True, outline_color='orange', b_scale=0.5):
    boundary_mask2d = get_section_boundary(mask, axis=axis, v=1, c=sectionX)
    sh, sw = boundary_mask2d.shape[:2]
    if ax is None:
        fig, ax = plt.subplots()
        brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)
        im = np.ones((sh, sw, 4), dtype=np.uint8) * 255
        im[~brain_mask2d] = 0#1

    # show boundary
    b_indices = np.where(boundary_mask2d)
    ax.scatter(b_indices[1], b_indices[0], s=b_scale, c='black', alpha=0.5, edgecolors='none')
    # intra-brain regions

    if with_outline:
        outline_mask2d = get_brain_outline2d(mask, axis=axis, v=1)
        o_indices = np.where(outline_mask2d)
        ax.scatter(o_indices[1], o_indices[0], s=1.0, c=outline_color, alpha=1.0, edgecolors='none')

    if ax is None:
        return fig, ax
    else:
        return ax

def process_mip(mip, mask, sectionX=None, axis=0, figname='temp.png', mode='composite', with_outline=True, outline_color='orange', pt_scale=2, b_scale=0.5):
    # get the mask
    brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)

    #if axis==1: cv2.imwrite('temp.png', mip); sys.exit()
    im = np.ones((mip.shape[0], mip.shape[1], 4), dtype=np.uint8) * 255
    # default size is 6.4 x 4.8
    scale = np.sqrt(np.prod(mip.shape[:2]) / 456 / 320)
    wi, hi = np.round(6.4 * scale, 2), np.round(4.8 * scale, 2)

    fig, ax = plt.subplots(figsize=(wi, hi))
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)

    canvas = FigureCanvas(fig)
    im = ax.imshow(im)
    fig.patch.set_visible(False)
    ax.axis('off')

    bg_mask = mip.sum(axis=-1) == 0
    fg_mask = ~bg_mask
    fg_indices = np.where(fg_mask)
    if mode == 'composite':
        fg_values = mip[fg_indices] / 255.
        cmap = None
    else:
        fg_values = mip[fg_indices][:,0] / 255.
        cmap = 'coolwarm'

    if len(fg_indices[0]) > 0:
        ax.scatter(fg_indices[1], fg_indices[0], c=fg_values, s=pt_scale, edgecolors='none', cmap=cmap)
    plot_section_outline(mask, axis=axis, sectionX=sectionX, ax=ax, with_outline=with_outline, outline_color=outline_color, b_scale=b_scale)

    plt.savefig(figname, dpi=300)
    plt.close('all')

    #canvas.draw()       # draw the canvas, cache the renderer
    #img_buffer = canvas.tostring_rgb()
    #out = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    #return out

def get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex, axids=(2,), thickX2=20, disp_right_hemi=False):
    df, feat_names = process_features(mefile)

    c = len(feat_names)
    zdim, ydim, xdim = shape3d
    zdim2, ydim2, xdim2 = zdim//2, ydim//2, xdim//2
    memap = np.zeros((zdim, ydim, xdim, c), dtype=np.uint8)
    xyz = np.floor(df[['soma_x', 'soma_y', 'soma_z']].to_numpy()).astype(np.int32)
    # normalize to uint8
    fvalues = df[feat_names]
    fmin, fmax = fvalues.min(), fvalues.max()
    fvalues = ((fvalues - fmin) / (fmax - fmin) * 255).to_numpy()
    if histeq:
        for i in range(fvalues.shape[1]):
            fvalues[:,i] = image_histeq(fvalues[:,i])[0]

    if flip_to_left:
        # flip z-dimension, so that to aggregate the information to left or right hemisphere
        right_hemi_mask = xyz[:,2] < zdim2
        xyz[:,2][right_hemi_mask] = zdim - xyz[:,2][right_hemi_mask]
        # I would also like to show the right hemisphere
        if disp_right_hemi:
            xyz2 = xyz.copy()
            xyz2[:,2] = zdim - xyz2[:,2]
            # concat
            xyz = np.vstack((xyz, xyz2))
            # also for the values
            fvalues = np.vstack((fvalues, fvalues))

    debug = False
    if debug: #visualize the distribution of features
        g = sns.histplot(data=fvalues, kde=True)
        plt.savefig('fvalues_distr_histeq.png', dpi=300)
        plt.close('all')

    if mode == 'composite':
        memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues
    else:
        memap[xyz[:,2], xyz[:,1], xyz[:,0]] = fvalues[:,findex].reshape(-1,1)

    # keep only values near the section plane
    mips = []
    for axid in axids:
        print(f'--> Processing axis: {axid}')
        cur_mips = []
        for sid in range(thickX2, shape3d[axid], 2*thickX2):
            cur_memap = memap.copy()
            if thickX2 != -1:
                if axid == 0:
                    cur_memap[:sid-thickX2] = 0
                    cur_memap[sid+thickX2:] = 0
                elif axid == 1:
                    cur_memap[:,:sid-thickX2] = 0
                    cur_memap[:,sid+thickX2:] = 0
                else:
                    cur_memap[:,:,:sid-thickX2] = 0
                    cur_memap[:,:,sid+thickX2:] = 0
            print(cur_memap.mean(), cur_memap.std())

            mip = get_mip_image(cur_memap, axid)
            cur_mips.append(mip)
        mips.append(cur_mips)
    return mips

def generate_me_maps(mefile, outfile, histeq=True, flip_to_left=True, mode='composite', findex=0, fmt='svg', axids=(2,)):
    '''
    @param mefile:          file containing microenviron features
    @param outfile:         prefix of output file
    @param histeq:          Whether or not to use histeq to equalize the feature values
    @param flip_to_left:    whether map points at the right hemisphere to left hemisphere
    @param mode:            [composite]: show 3 features; otherwise separate feature
    @param findex:          index of feature to display
    '''
    if mode != 'composite':
        fname = __MAP_FEATS__[findex]
        prefix = f'{outfile}_{fname}'
    else:
        prefix = f'{outfile}'

    mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
    shape3d = mask.shape
    thickX2 = 20
    mips = get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex, axids=axids, thickX2=thickX2)
    for axid, cur_mips in zip(axids, mips):
        for imip, mip in enumerate(cur_mips):
            figname = f'{prefix}_mip{axid}_{imip:02d}.{fmt}'
            sectionX = thickX2 * (2 * imip + 1)
            process_mip(mip, mask, axis=axid, figname=figname, mode=mode, sectionX=sectionX, with_outline=False)
            if not figname.endswith('svg'):
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
                    # concatenate with the sectional atlas
                    outscale = 3
                    atlas2d = colorize_atlas2d_cv2(axid, sectionX, outscale=outscale, annot=True, fmt='png')
                    #
                    atlas2d = cv2.resize(atlas2d, img.shape[:2][::-1])
                    r2 = img.shape[1]//2
                    img[:,r2:,:3] = atlas2d[:,r2:]

                cv2.imwrite(figname, img)

def colorize_atlas2d_cv2(axid=2, sectionX=420, outscale=3, annot=False, fmt='svg'):
    mask = load_image(MASK_CCF25_FILE)
    ana_dict = parse_ana_tree()

    boundaries = get_section_boundary(mask, axis=axid, c=sectionX, v=1)
    section = np.take(mask, sectionX, axid)
    out = np.ones((*section.shape, 3), dtype=np.uint8) * 255
    values = np.unique(section)
    print(f'Dimension of axis={axid} is: {section.shape}, with {len(values)-1} regions')

    if annot:
        centers = []
        rnames = []
        c2 = out.shape[0] // 2
        right_mask = boundaries.copy()
        right_mask.fill(False)
        right_mask[:c2] = True
        for v in values:
            if v == 0: continue
            rname = ana_dict[v]['acronym']

            # center of regions,
            cur_mask = section == v
            out[:,:,:3][cur_mask] = ana_dict[v]['rgb_triplet']

            if rname in ['root', 'fiber tracts']:   # no texting is necessary
                continue
            if axid != 0:
                cur_mask = cur_mask & right_mask #only left hemisphere
            cc = cv2.connectedComponents(cur_mask.astype(np.uint8))
            for i in range(cc[0] - 1):
                cur_mask = cc[1] == (i+1)
                if cur_mask.sum() < 5:
                    continue
                indices = np.where(cur_mask)
                xmean = (indices[0].min() + indices[0].max()) // 2
                ymean = int(np.median(indices[1][indices[0] == xmean]))
                centers.append((xmean, ymean))
                rnames.append(rname)
    else:
        for v in values:
            if v == 0: continue
            cur_mask = section == v
            out[:,:,:3][cur_mask] = ana_dict[v]['rgb_triplet']
    # mixing with boudnary
    alpha = 0.2
    out[:,:,:3][boundaries] = (0 * alpha + out[boundaries][:,:3] * (1 - alpha)).astype(np.uint8)
    #out[:,:,3][boundaries] = int(alpha * 255)

    
    figname = f'atlas_axis{axid}.{fmt}'
    if outscale != 1:
        out = cv2.resize(out, (0,0), fx=outscale, fy=outscale, interpolation=cv2.INTER_CUBIC)
    # we would like to rotate the image, so that it can be better visualized
    if axid != 0:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

    so1, so2 = out.shape[:2]
    # annotation if required
    if annot:
        figname = f'atlas_axis{axid}_section{sectionX}_annot.{fmt}'
        shift = 20
        for center, rn in zip(centers, rnames):
            sx, sy = center[1]*outscale, center[0]*outscale
            if axid != 0:
                # rotate accordingly
                new_center = (so2-sy-shift, sx)
            else:
                new_center = (sx-shift, sy)
            cv2.putText(out, rn, new_center, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1)

    if figname.endswith('svg'):
        # save to `svg` vectorized file, using plt
        fig, ax = plt.subplots()
        ax.imshow(out)
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.savefig(figname, dpi=300)
        plt.close('all')
    else:
        cv2.imwrite(figname, out)

    return out

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
    df = pd.read_csv(mefile, index_col=0)
    keys = [f'{key}' for key in __MAP_FEATS__]
    rkey = 'region_name'
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
    #print(dfr.shape); sys.exit()
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
        
        figname = f'{out_prefix}_section{sid:03d}_ion.png'
        process_mip(mip, sub_mask, axis=axid, figname=figname, sectionX=sid, with_outline=False, pt_scale=5, b_scale=0.5)

        section_mask = sub_mask[:,:,sid]
        
        if 0:
            # also save the mask
            img_mask = np.zeros((*section_mask.shape, 4), dtype=np.uint8)
            for rid, cname in cmap.items():
                rgba = [int(c*255) for c in colors.to_rgba(cname[0])]
                #import ipdb; ipdb.set_trace()
                rgba[:3] = rgba[:3][::-1]
                img_mask[section_mask == rid] = rgba
            if axid != 0:
                img_mask = cv2.rotate(img_mask, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(f'mask_{figname}', img_mask)

            # generate the color legend
            xa = np.random.random((5))
            xb = np.random.random((5))
            for rid, rcolor in cmap.items():
                plt.scatter(xa, xb, c=rcolor[0], label=rcolor[-1])
            plt.legend(frameon=False, ncol=4, handletextpad=0.2)
            plt.savefig('color_legend.png', dpi=300)
            plt.close()


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
    mefile = './ION_HIP/lm_features_d28_dendrites.csv'
    mapfile = 'ion_local'
    scale = 25.
    flip_to_left = True
    mode = 'composite'
    axids = (2,)
    findex = 0
    fmt = 'png'



    if 1:
        swcdir = '/data/lyf/data/200k_v2/cropped_100um_resampled2um'

        #rname = ['ACAv2/3', 'AIv2/3', 'GU2/3', 'MOp2/3', 'MOs2/3', 'ORBl2/3', 'ORBm2/3', 'ORBvl2/3', 'PL2/3', 'RSPv2/3', 'SSp-m2/3', 'SSp-n2/3']
        rname = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
        plot_region_feature_sections(mefile, rname)
   
