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
from mpl_toolkits.mplot3d import Axes3D
from fil_finder import FilFinder2D
import astropy.units as u
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, ANATOMY_TREE_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, \
                                get_brain_mask2d, get_section_boundary, detect_edges2d
from anatomy.anatomy_core import parse_ana_tree
from plotters.neurite_arbors import NeuriteArbors

from config import mRMR_f3 as __MAP_FEATS__

if __name__ == '__main__':
    # customize the fonts
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.weight'] = 'light'


# features selected by mRMR
#__MAP_FEATS__ = ('Length', 'AverageFragmentation', 'AverageContraction')   # original one

def process_features(mefile, scale=25., with_comment=False):
    if with_comment:
        df = pd.read_csv(mefile, index_col=0, comment='#')
    else:
        df = pd.read_csv(mefile, index_col=0)
    df.drop(list(__MAP_FEATS__), axis=1, inplace=True)

    mapper = {}
    for mf in __MAP_FEATS__:
        mapper[f'{mf}_me'] = mf
    df.rename(columns=mapper, inplace=True)
    # We would like to use tortuosity, which is  opposite of contraction
    #df.loc[:, 'AverageContraction'] = 1 - df['AverageContraction']

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

def find_regional_representative(mefile, region='IC', swcdir='', color='magenta'):
    random.seed(1024)
    df = pd.read_csv(mefile, index_col=0)
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    #import ipdb; ipdb.set_trace()
    tmp = df[keys]
    dfr = df.copy()
    dfr[keys] = (tmp - tmp.mean()) / tmp.std()
    # keep only neurons from the target region
    rmask = dfr.region_name_r316 == region
    dfr = dfr[rmask][keys]
    print(f'Number of neurons in region {region}: {dfr.shape[0]}')
    medians = dfr.median()
    # find out the neurons with closest features
    dists2m = np.linalg.norm(dfr - medians, axis=1)
    min_id = dists2m.argmin()
    min_dist = dists2m[min_id]
    min_name = dfr.index[min_id]
    min_brain = df.loc[min_name, 'brain_id']
    print(f'The neuron {min_name} has distance {min_dist:.4f} to the median of the region {region}')
    print(df[rmask][keys].iloc[min_id], min_brain)
    
    plot_morphology = True
    if plot_morphology:
        if 0:
            # plot the neurons
            nsamples = 50
            for swc_name in random.sample(list(df[rmask].index), nsamples):
                brain_id = df.loc[swc_name, 'brain_id']
                swcfile = os.path.join(swcdir, str(brain_id), f'{swc_name}_stps.swc')
                na = NeuriteArbors(swcfile)
                out_name = f'{region}_{brain_id}_{swc_name}'
                na.plot_morph_mip(type_id=None, color=color, figname=out_name, out_dir='.', show_name=False)
        if 1:   # to match previous selection
            #brain_id, swc_name = 191797, '6987_19683_7370'
            brain_id, swc_name = 201584, '8378_11445_10616'
            swcfile = os.path.join(swcdir, str(brain_id), f'{swc_name}_stps.swc')
            na = NeuriteArbors(swcfile)
            out_name  = f'{region}_{brain_id}_{swc_name}'
            na.plot_morph_mip(type_id=None, color=color, figname=out_name, out_dir='.', show_name=False)

def plot_inter_regional_features(mefile, regions=('IC', 'SIM')):
    df = pd.read_csv(mefile, index_col=0)
    keys = ['region_name_r316'] + [f'{key}_me' for key in __MAP_FEATS__]
    dfr = df[keys][df['region_name_r316'].isin(regions)]
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # plot
    n1, n2, n3 = keys[-3:]
    dfr1 = dfr[dfr['region_name_r316'] == regions[0]]
    dfr2 = dfr[dfr['region_name_r316'] == regions[1]]
    sc1 = ax.scatter(dfr1[n3], dfr1[n2], dfr1[n1]/1000, s=12, c='magenta', marker='o', alpha=.75, label=regions[0])
    sc2 = ax.scatter(dfr2[n3], dfr2[n2], dfr2[n1]/1000, s=12, c='cyan', marker='o', alpha=1., label=regions[1])

    label_size = 22
    ax.set_xlabel('Fragmentation', fontsize=label_size, labelpad=8)
    ax.set_ylabel('Straightness', fontsize=label_size, labelpad=10)
    ax.set_zlabel('Total Length (mm)', fontsize=label_size, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yticks([0.8,0.84,0.88,0.92,0.96])

    # legend
    plt.legend(bbox_to_anchor=(0.6,0.6), fontsize=label_size, markerscale=3., handletextpad=0.2, frameon=False)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    ax.set_box_aspect(None, zoom=0.85)  # to avoid Z label cutoff
    #hide the gridline
    ax.grid(False)
    elev, azim = 30, -15
    ax.view_init(elev, azim)
    # save
    plt.savefig("IC_SIM_features.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_MOB_features(mefile, rname='MOB', r316=False):
    df = pd.read_csv(mefile, index_col=0)
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    if r316:
        rkey = 'region_name_r316'
    else:
        rkey = 'region_name_r671'

    if type(rname) is list:
        dfr = df[keys][df[rkey].isin(rname)]
        out_prefix = 'tmp'
    else:
        dfr = df[keys][df[rkey] == rname]
        out_prefix = rname

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0] / 255.
    dfc[dfc > 1] = 1.
        
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # plot
    n1, n2, n3 = keys[-3:]
    print(n1, n2, n3)
    sc = ax.scatter(dfr[n3], dfr[n2], dfr[n1]/1000, s=10, c=dfc.values, marker='o', alpha=.75)
    label_size = 22
    ax.set_xlabel('Fragmentation', fontsize=label_size, labelpad=10)
    ax.set_ylabel('Straightness', fontsize=label_size, labelpad=10)
    ax.set_zlabel('Total Length (mm)', fontsize=label_size, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=14)

    # legend
    plt.legend(bbox_to_anchor=(0.6,0.6), fontsize=label_size, markerscale=3., handletextpad=0.2)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    ax.set_box_aspect(None, zoom=0.85)  # to avoid Z label cutoff
    if rname in ['MOB', 'ACB']:
        ax.set_zlim3d(0.5, 2)
    ax.get_legend().remove()
    # Hide grid lines
    ax.grid(False)
    if rname == 'ACB':
        ax.view_init(30, 30)

    # save
    if '/' in rname:
        rname = rname.replace('/', '_')
    plt.savefig(f"{out_prefix}_features.png", bbox_inches='tight')
    plt.close()

def plot_region_feature_in_ccf_space(mefile, rname='MOB', r316=False, flipLR=True):
    df = pd.read_csv(mefile, index_col=0)
    keys = [f'{key}_me' for key in __MAP_FEATS__]
    if r316:
        rkey = 'region_name_r316'
    else:
        rkey = 'region_name_r671'

    if type(rname) is list:
        sel_mask = df[rkey].isin(rname)
        out_prefix = 'tmp'
    else:
        sel_mask = df[rkey] == rname
        out_prefix = rname
    
    dfr = df[keys][sel_mask]
    coords = df[['soma_x', 'soma_y', 'soma_z']][sel_mask].values / 1000
    if flipLR:
        zdim = 456 * 25. / 1000
        right = np.nonzero(coords[:,2] > zdim/2)[0]
        coords[right, 2] = zdim - coords[right, 2]

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0] / 255.
    dfc[dfc > 1] = 1.
        
    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # plot
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=10, c=dfc.values, marker='o', alpha=.75)
    label_size = 22
    ax.set_xlabel('AP axis (mm)', fontsize=label_size, labelpad=10)
    ax.set_ylabel('LR axis (mm)', fontsize=label_size, labelpad=10)
    ax.set_zlabel('DV axis (mm)', fontsize=label_size, labelpad=10)

    ax.tick_params(axis='both', which='major', labelsize=14)

    # legend
    plt.legend(bbox_to_anchor=(0.6,0.6), fontsize=label_size, markerscale=3., handletextpad=0.2)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    ax.set_box_aspect(None, zoom=0.85)  # to avoid Z label cutoff
    ax.get_legend().remove()
    # Hide grid lines
    ax.grid(False)
    #ax.view_init(0, 30)

    # save
    if '/' in out_prefix:
        out_prefix = out_prefix.replace('/', '_')
    plt.savefig(f"{out_prefix}_features_ccf.png", bbox_inches='tight')
    plt.close()

def plot_region_feature_sections(mefile, rname='MOB', r316=False, flipLR=True, thickX2=10):
    df = pd.read_csv(mefile, index_col=0)
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

    # We handling the coloring
    dfc = dfr.copy()
    for i in range(3):
        tmp = dfc.iloc[:,i]
        dfc.iloc[:,i] = image_histeq(tmp.values)[0]
    dfc[dfc > 255] = 255

    # get the boundary of region
    nzcoords = rmask.nonzero()
    nzcoords_t = np.array(nzcoords).transpose()
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)
    sub_mask = rmask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    memap = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)
    
    coords_s = np.floor(coords * 40).astype(int)
    memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = dfc.values

    mips = []
    shape3d = mask.shape
    axid = 2
    for sid in range(0, xmax-xmin-thickX2-1, thickX2*2):
        sid = sid + thickX2
        cur_memap = memap.copy()
        cur_memap[:,:,:sid-thickX2] = 0
        cur_memap[:,:,sid+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())

        mip = get_mip_image(cur_memap, axid)
        
        figname = f'{out_prefix}_section{sid:03d}.png'
        print(mip.shape, sub_mask.shape)
        process_mip(mip, sub_mask, axis=axid, figname=figname, sectionX=sid, with_outline=False, pt_scale=5, b_scale=0.5)
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
 
    

def plot_parcellations(parc_file, ccf_tree_file=ANATOMY_TREE_FILE, ccf_atlas_file=MASK_CCF25_FILE):
    thickX2 = 20
    axid = 2
    parc = load_image(parc_file)
    # flip
    zdim2 = parc.shape[0] // 2 
    parc[:zdim2] = parc[zdim2:][::-1]
    ana_tree = parse_ana_tree(ccf_tree_file)
    ccf25 = load_image(ccf_atlas_file)
    shape3d = parc.shape
    prefix = os.path.splitext(os.path.split(parc_file)[-1])[0]
    for isid, sid in enumerate(range(thickX2, shape3d[axid], 2*thickX2)):
        figname = f'{prefix}_{isid:02d}.png'
        section = np.take(parc, sid, 2)
        ccf25s = np.take(ccf25, sid, 2)
        vuniq = np.unique(ccf25s)
        # coloring with CCF color scheme
        out = np.ones((*section.shape, 4), dtype=np.uint8) * 255
        out3 = out[:,:,:3]
        for vi in vuniq:
            if vi == 0: continue
            rmask = ccf25s == vi
            #print(rmask.sum())
            out3[rmask] = ana_tree[vi]['rgb_triplet']

        # draw the sub-parcellation
        parc_edges = detect_edges2d(section)
        ccf25_edges = detect_edges2d(ccf25s)
        extra_edges = parc_edges ^ ccf25_edges
        extra_edges[:zdim2] = extra_edges[zdim2:][::-1]
        out[extra_edges] = (0,0,255,255)
        # draw the original ccf outline
        out[ccf25_edges] = (0,0,0,128)
        # zeroing the background
        # rotate
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(figname, out)
        print()


    

if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    mapfile = 'microenviron_map'
    scale = 25.
    flip_to_left = True
    mode = 'composite'
    axids = (2,)
    findex = 0
    fmt = 'png'

    if 1:
        generate_me_maps(mefile, outfile=mapfile, flip_to_left=flip_to_left, mode=mode, findex=findex, fmt=fmt, axids=axids)

    if 0:
        mefile = './data/mefeatures_100K_with_PCAfeatures3.csv'
        swcdir = '/data/lyf/data/200k_v2/cropped_100um_resampled2um/'
        region = 'IC'
        if region == 'IC':
            color = 'black' #'magenta'
        elif region == 'SIM':
            color = 'black' #'cyan'

        #find_regional_representative(mefile, region=region, swcdir=swcdir, color=color)
        #plot_inter_regional_features(mefile)
        rname = ['ACAv2/3', 'AIv2/3', 'GU2/3', 'MOp2/3', 'MOs2/3', 'ORBl2/3', 'ORBm2/3', 'ORBvl2/3', 'PL2/3', 'RSPv2/3', 'SSp-m2/3', 'SSp-n2/3']
        #rname = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
        #plot_MOB_features(mefile, 'MOB')
        #plot_region_feature_in_ccf_space(mefile, 'CA1')
        #plot_region_feature_sections(mefile, 'CA1')
   
    if 0:
        parc_file = 'intermediate_data/parc_r671_full.nrrd'
        plot_parcellations(parc_file)
