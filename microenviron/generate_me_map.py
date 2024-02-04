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
from fil_finder import FilFinder2D
import astropy.units as u
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, get_brain_mask2d, get_section_boundary
from anatomy.anatomy_core import parse_ana_tree


# features selected by mRMR
__MAP_FEATS__ = ('Length', 'AverageFragmentation', 'AverageContraction')

def process_features(mefile, scale=25.):
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

def plot_section_outline(mask, axis=0, sectionX=None, ax=None, with_outline=True, outline_color='orange'):
    boundary_mask2d = get_section_boundary(mask, axis=axis, v=1, c=sectionX)
    sh, sw = boundary_mask2d.shape[:2]
    if ax is None:
        fig, ax = plt.subplots()
        brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)
        im = np.ones((sh, sw, 4), dtype=np.uint8) * 255
        im[~brain_mask2d] = 0#1

    # show boundary
    b_indices = np.where(boundary_mask2d)
    ax.scatter(b_indices[1], b_indices[0], s=0.5, c='black', alpha=0.5, edgecolors='none')
    # intra-brain regions
        
    if with_outline:
        outline_mask2d = get_brain_outline2d(mask, axis=axis, v=1)
        o_indices = np.where(outline_mask2d)
        ax.scatter(o_indices[1], o_indices[0], s=1.0, c=outline_color, alpha=1.0, edgecolors='none')
    
    if ax is None:
        return fig, ax
    else:
        return ax
    

def process_mip(mip, mask, sectionX=None, axis=0, figname='temp.png', mode='composite', with_outline=True, outline_color='orange'):
    # get the mask
    brain_mask2d = get_brain_mask2d(mask, axis=axis, v=1)

    #if axis==1: cv2.imwrite('temp.png', mip); sys.exit()
    im = np.ones((mip.shape[0], mip.shape[1], 4), dtype=np.uint8) * 255
    im[~brain_mask2d] = 1 # should be 1 for processing
    
    fig, ax = plt.subplots()
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
        ax.scatter(fg_indices[1], fg_indices[0], c=fg_values, s=3, edgecolors='none', cmap=cmap)
    plot_section_outline(mask, axis=axis, sectionX=sectionX, ax=ax, with_outline=with_outline, outline_color=outline_color)

    plt.savefig(figname, dpi=300)
    plt.close('all')

    #canvas.draw()       # draw the canvas, cache the renderer
    #img_buffer = canvas.tostring_rgb()
    #out = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    #return out

def get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex):
    df, feat_names = process_features(mefile)
    
    c = len(feat_names)
    zdim, ydim, xdim = shape3d
    zdim2, ydim2, xdim2 = zdim//2, ydim//2, xdim//2
    memap = np.zeros((zdim, ydim, xdim, c), dtype=np.uint8)
    xyz = np.floor(df[['soma_x', 'soma_y', 'soma_z']].to_numpy()).astype(np.int32)
    if flip_to_left:
        # flip z-dimension, so that to aggregate the information to left or right hemisphere
        right_hemi_mask = xyz[:,2] < zdim2
        xyz[:,2][right_hemi_mask] = zdim - xyz[:,2][right_hemi_mask]

    # normalize to uint8
    fvalues = df[feat_names]
    fmin, fmax = fvalues.min(), fvalues.max()
    fvalues = ((fvalues - fmin) / (fmax - fmin) * 255).to_numpy()
    if histeq:
        for i in range(fvalues.shape[1]):
            fvalues[:,i] = image_histeq(fvalues[:,i])[0]
    
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
    thickX2 = 20
    mips = []
    for axid in range(3):
        print(f'--> Processing axis: {axid}')
        cur_memap = memap.copy()
        print(cur_memap.mean(), cur_memap.std())
        if thickX2 != -1:
            if axid == 0:
                cur_memap[:zdim2-thickX2] = 0
                cur_memap[zdim2+thickX2:] = 0
            elif axid == 1:
                cur_memap[:,:ydim2-thickX2] = 0
                cur_memap[:,ydim2+thickX2:] = 0
            else:
                cur_memap[:,:,:xdim2-thickX2] = 0
                cur_memap[:,:,xdim2+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())
        
        mip = get_mip_image(cur_memap, axid)
        mips.append(mip)
    return mips

def generate_me_maps(mefile, outfile, histeq=True, flip_to_left=True, mode='composite', findex=0, fmt='svg'):
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
    mips = get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex)
    for axid, mip in enumerate(mips):
        figname = f'{prefix}_mip{axid}.{fmt}'
        process_mip(mip, mask, axis=axid, figname=figname, mode=mode)
        if not figname.endswith('svg'):
            # load and remove the zero-alpha block
            img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
            wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
            ws, we = wnz[0], wnz[-1]
            hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
            hs, he = hnz[0], hnz[-1]
            img = img[hs:he+1, ws:we+1]
            if axid != 0:   # rotate 90
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # set the alpha of non-brain region as 0
            img[img[:,:,-1] == 1] = 0
            cv2.imwrite(figname, img)
        
def plot_left_right_corr(mefile, outfile, histeq=True, mode='composite', findex=0):
    if mode != 'composite':
        fname = __MAP_FEATS__[findex]
        prefix = f'{outfile}_{fname}'
    else:
        prefix = f'{outfile}'

    def customized_func(x, a):
        return a*x

    def customized_line(xkey, ykey, color=None, label=None, **kwargs):
        # we should use `y = ax` to fit our data.
        ax = plt.gca()
        popt, pcov = curve_fit(customized_func, xkey, ykey)
        xmin = min(xkey.min(), ykey.min())
        xmax = max(xkey.max(), ykey.max())
        xdata = np.arange(xmin, xmax)
        ydata = popt[0] * xdata
        ax.plot(xdata, ydata, linewidth=3, color=color, label=label)
        # estimate the R_squared
        #residuals = ykey - customized_func(xkey, *popt)
        #ss_res = np.sum(residuals**2)
        #ss_tot = np.sum((ykey - np.mean(ykey))**2)
        #r_squared = 1 - (ss_res / ss_tot)
        r_squared = r2_score(ykey, customized_func(xkey, *popt))

        # annotate
        ax.text(.03, .85, f'y={popt[0]:.2f}x\nR^2={r_squared:.2g}', 
                fontsize=20, transform=ax.transAxes, color=color)

    
    mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
    shape3d = mask.shape
    mips = get_me_mips(mefile, shape3d, False, False, mode, findex)
    xlabel = 'Value predicted \nfrom left-hemisphere'
    ylabel = 'Right-hemispheric value'
    data = []
    for axid, mip in enumerate(mips):
        if axid == 0: continue # no left/right difference
        s1, s2 = mip.shape[:2]
        sh = s1 // 2
        print(s1, s2, sh)
        fg_mask_left = mip.sum(axis=-1) != 0; fg_mask_left[:sh] = 0
        fg_mask_right = mip.sum(axis=-1) != 0; fg_mask_right[sh:] = 0
        vpreds = []
        vtrues = []
        lindices = np.where(fg_mask_left)
        rindices = np.where(fg_mask_right)
        rindices_m = (s1 - rindices[0], rindices[1])
        for i in range(mip.shape[2]):
            lvalues = mip[:,:,i][lindices]
            interp = LinearNDInterpolator(np.transpose(lindices), lvalues)
            vpreds.append(interp(*rindices_m))
            vtrues.append(mip[:,:,i][rindices])
        values = np.vstack((np.hstack(vpreds), np.hstack(vtrues)))
        fnames = []
        for mf in __MAP_FEATS__:
            for j in range(rindices[0].shape[0]):
                fnames.append(mf)
        cur_data = np.array([*values, fnames]).transpose()
        data.append(cur_data)

    size1, size2 = data[0].shape[0], data[1].shape[0]
    views = ['Axial' for i in range(size1)] + ['Coronal' for i in range(size2)]
    data = np.vstack(data)
    figname = f'{prefix}_left_right.png'
    df = pd.DataFrame(data, columns=[xlabel, ylabel, 'feature']).astype({xlabel: float, ylabel: float})
    df['view'] = views
    # remove nan values
    df = df[~df[xlabel].isna()]
    print(df.shape)

    sns.set_theme(style='white', font_scale=1.8)
    g = sns.lmplot(data=df, x=xlabel, y=ylabel, col='feature', 
                   row='view', fit_reg=False,
                   scatter_kws={'s':5}, 
                   facet_kws={'despine': False, 'margin_titles': True})
    g.map(customized_line, xlabel, ylabel, color='magenta')
    g.set(xlim=(0,255), ylim=(0,255), aspect='equal')
    for i, ax_list in enumerate(g.axes):
        for j, ax in enumerate(ax_list):
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.xaxis.set_tick_params(width=2, direction='in')
            ax.yaxis.set_tick_params(width=2, direction='in')
            if i == 0:
                #title = ax.get_title().split(' = ')[1]
                if j == 0:
                    title = 'Straightness'
                elif j == 1:
                    title = 'Hausdorff Dimension'
                elif j == 2:
                    title = '%Variance of PC_3'
                ax.set_title(title)

    g.figure.subplots_adjust(wspace=-0.1, hspace=0)
    plt.savefig(figname, dpi=300)
    plt.close('all')
        
def colorize_atlas2d_cv2(outscale=3, annot=False, fmt='svg'):
    mask = load_image(MASK_CCF25_FILE)
    ana_dict = parse_ana_tree()
    for axid in range(3):
        boundaries = get_section_boundary(mask, axis=axid, c=None, v=1)
        c = mask.shape[axid] // 2
        section = np.take(mask, c, axid)
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
            figname = f'atlas_axis{axid}_annot.{fmt}'
            for center, rn in zip(centers, rnames):
                sx, sy = center[1]*outscale, center[0]*outscale
                if axid != 0:
                    # rotate accordingly
                    new_center = (so2-sy, sx)
                else:
                    new_center = (sx, sy)
                cv2.putText(out, rn, new_center, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)

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

def sectional_dsmatrix(mefile, outfile, histeq=True, flip_to_left=True, mode='composite', findex=0):
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
    mips = get_me_mips(mefile, shape3d, histeq, flip_to_left, mode, findex)
    ana_dict = parse_ana_tree(keyname='id')
    for axid, mip in enumerate(mips):
        if histeq:
            out_file = f'{prefix}_mip{axid}_histeq.csv'
        else:
            out_file = f'{prefix}_mip{axid}.csv'
        if axid != 1: continue
        mask2d = np.take(mask, mask.shape[axid]//2, axid)
        nz_mask = (mip.sum(axis=-1) > 0) & (mask2d > 0)
        mask2d[~nz_mask] = 0
        # calculate ds
        mip = mip / 255.
        df = pd.DataFrame(mip[nz_mask], columns=['f1', 'f2', 'f3'])
        df['rid'] = mask2d[nz_mask]
        df['rname'] = [ana_dict[idx]['acronym'] for idx in df['rid']]
        cxs, cys = [], []
        for rid in df.rid:
            cur_reg = np.nonzero(mask2d == rid)
            cx = cur_reg[0].mean()
            cy = cur_reg[1].mean()
            cxs.append(cx)
            cys.append(cy)
        df['xcenter'] = cxs
        df['ycenter'] = cys
        df.to_csv(out_file)

def plot_me_dsmatrix(feat_file, feat_file_histeq, axid, min_num_samples=10):
    df = pd.read_csv(feat_file, index_col=0)
    df_histeq = pd.read_csv(feat_file_histeq, index_col=0)
    # filter by number
    regs, counts = np.unique(df['rname'], return_counts=True)
    # remove fiber tracts or similar
    non_tracts = np.array([not rname.islower() for rname in regs])
    keep_mask = (counts >= min_num_samples) & non_tracts
    keep_regs = regs[keep_mask]
    keep_counts = counts[keep_mask]
    df = df[df['rname'].isin(keep_regs)]
    df_histeq = df_histeq[df_histeq['rname'].isin(keep_regs)]
    '''df_feat = df.iloc[:,:3]
    corr = pd.DataFrame(distance_matrix(df_feat, df_feat), index=df.rname, columns=df.rname)
    #corr = df_feat.transpose().corr()
    #corr = corr.rename(columns=dict(zip(corr.columns, df.rname))).rename(index=dict(zip(corr.index, df.rname)))
    print(corr.shape)
    mcorr = corr.groupby(by=corr.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
    mcorr = mcorr.groupby(by=mcorr.index, axis=0).apply(lambda g: g.mean(axis=0) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
    # plot
    structs = []
    r2s_dict = dict(zip(df.rname, df.struct))
    for rn in mcorr.index:
        structs.append(r2s_dict[rn])
    lut = dict(zip(np.unique(structs), "rbgm"))
    row_colors = [lut[s] for s in structs]
    cm = sns.clustermap(mcorr, cmap='coolwarm_r', 
                       xticklabels=1, yticklabels=1,
                       row_colors=row_colors)
    plt.savefig('region_corr.png', dpi=200)
    plt.close('all')
    '''

    # do clustering
    fmean = df.groupby('rid').mean()
    fstd = df.groupby('rid').std()
    fmedian = df_histeq.groupby('rid').median()
    fmerge = fmean.merge(fstd, how='inner', on='rid').merge(fmedian, how='inner', on='rid')[['f1_x', 'f2_x', 'f3_x', 'f1_y', 'f2_y', 'f3_y', 'f1', 'f2', 'f3']]
    nclass = 6
    palette = {
        0: (102,255,102),
        1: (255,102,102),
        2: (255,255,102),
        3: (102,255,255),
        4: (102,102,255),
        5: (178,102,255)
    }
    kmeans = KMeans(n_clusters=nclass)
    kmeans.fit(fmerge[['f1_x', 'f2_x', 'f3_x', 'f1_y', 'f2_y', 'f3_y']])
    mask = load_image(MASK_CCF25_FILE)
    mask2d = np.take(mask, mask.shape[axid]//2, axid)
    mip = np.zeros((*mask2d.shape[:2], 3), dtype=mask2d.dtype)
    for rid, cid in zip(fmerge.index, kmeans.labels_):
        mip[mask2d == rid] = palette[cid]
    process_mip(mip, mask, sectionX=None, axis=axid, figname='feature_classes.png', mode='composite')
    
    # plot the evolution of features along path
    paths = {
        0: ['OLF', 'MOB', 'ORBvl1', 'ORBvl2/3', 'ORBl5', 'AId5', 'MOp5', 'GU5', 'SSp-m5', 
            'SSs5', 'VISC5', 'TEa5', 'ECT5', 'ENTl5', 'ENTm5', 'PAR', 'PRE', 'SUB', 'CA1',
            'CA3', 'DG-mo'],
        1: ['SSs1', 'SSs2/3', 'SSs4', 'SSs5', 'SSs6a', 'CP', 'VPL', 'VPM', 'PO', 
            'CL', 'MD'],
        #2: ['STR', 'LSr', 'SF', 'PVT', 'IMD', 'PF', 'MRN', 'MB']
        2: ['ILA6a', 'DP', 'STR', 'LSr', 'SF', 'RT', 'CA3', 'CA1', 'SUB', 
            'PAR', 'PRE', 'DG-mo']
    }
    fmerge2 = fmerge.rename(index=dict(zip(df.rid, df.rname)))
    #colors_f = ['orange', 'mediumblue', 'lime']
    colors_f = ['red', 'green', 'blue']
    for i, path in paths.items():
        cur_data = fmerge2.loc[path][['f1_x', 'f2_x', 'f3_x', 'f1_y', 'f2_y', 'f3_y']]
        cur_median = fmerge2.loc[path][['f1', 'f2', 'f3']]
        xp = np.arange(cur_data.shape[0])
        for jj in range(3):
            label = __MAP_FEATS__[jj]
            if label == 'pca_vr3':
                label = '%Variance PC_3'
            elif label == 'AverageContraction':
                label = 'Straightness'
            elif label == 'HausdorffDimension':
                label = 'Hausdorff Dimension'
            #lower = cur_data.iloc[:,jj]-cur_data.iloc[:,jj+3]
            #upper = cur_data.iloc[:,jj]+cur_data.iloc[:,jj+3]
            #plt.fill_between(xp, lower, upper, color=colors_f[jj], alpha=0.2)
            #plt.errorbar(xp, cur_data.iloc[:,jj], yerr=cur_data.iloc[:,jj+3], 
            #            elinewidth=5, color=colors_f[jj], alpha=0.5)
            plt.plot(xp, cur_data.iloc[:, jj], 's-', color=colors_f[jj], label=label)
        
        fs = 13
        plt.xlim(xp[0]-1, xp[-1]+1)
        plt.ylim(0.1, 0.9)
        plt.xticks(xp, path, rotation=90, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xlabel(f'Region along Path{i+1}', fontsize=fs*1.5)
        plt.ylabel('Normalized feature value', fontsize=fs*1.5)
        ax = plt.gca()
        ax.xaxis.set_tick_params(width=2, direction='in')
        ax.yaxis.set_tick_params(width=2, direction='in')
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        # customized the colors of region
        for ixtick, xtick in enumerate(ax.get_xticklabels()):
            color = cur_median.iloc[ixtick].to_numpy()
            xtick.set_color(color)

        plt.legend(frameon=False)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(f'feature_along_path{i}.png', dpi=300)
        plt.close('all')


def feature_evolution_CP_radial(mefile, debug=True):
    mask = load_image(MASK_CCF25_FILE)
    shape3d = mask.shape
    cp_id = 672
    axid = 1
    left_axid = 0
    scale = .04
    
     # estimate the mip with features
    temp_file = 'temp.pkl'
    if os.path.exists(temp_file):
        with open(temp_file, 'rb') as fp:
            mips = pickle.load(fp)
    else:
        mips = get_me_mips(mefile, shape3d, False, True, 'composite', 0)
        with open(temp_file, 'wb') as fp:
            pickle.dump(mips, fp)

    mask2d = np.take(mask, shape3d[axid]//2, axid)

    #nrids = ['VL', 'LSr', 'STR', 'GPe', 'int', 'or', 'ar', 'aIv']
    nrids = [81, 258, 477, 1022, 6, 484682520, 484682524, 466]
    # mask for target regions
    cp_mask = mask2d == cp_id
    for i, v in enumerate(nrids):
        if i == 0:
            ngb = mask2d == v
        else:
            ngb = ngb | (mask2d == v)
    if left_axid == 0:
        ngb[:ngb.shape[0]//2] = 0
    elif left_axid == 1:
        ngb[:,:ngb.shape[1]//2] = 0
    # morphology closing to get better neighboring regions
    ngb = morphology.dilation(ngb, morphology.square(3))
    eks = 7
    ekernel = np.zeros((eks,eks)); ekernel[:,eks//2+1] = 1
    ngb = morphology.erosion(ngb, ekernel)
    # map all coordinates according the distance of neighboring points
    ngb_pts = np.stack(ngb.nonzero()).transpose()
    # fg points
    mip = mips[axid] / 255. # to 0-1
    cp_fg = (mip.sum(axis=-1) > 0) & cp_mask
    cp_fg_crds = np.stack(cp_fg.nonzero()).transpose()

    kdtree = KDTree(ngb_pts, leaf_size=2)
    dmin, imin = kdtree.query(cp_fg_crds, k=1)
    dmin *= scale
    df_cp = pd.DataFrame(np.hstack((dmin, mip[cp_fg])), columns=['Distance', *__MAP_FEATS__])
    sns.lmplot(data=df_cp, x='Distance', y='pca_vr3', aspect=1.4,
               robust=True, ci=95, n_boot=200,
               scatter_kws={'color':'lightsalmon', 's':10}, 
               line_kws={'color':'red'})

    fs = 14
    plt.xlim(0,70*scale)
    plt.ylim(0,0.9)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    #plt.xlabel('Distance along Path4 (\u03bcm)', fontsize=fs*1.5)
    plt.xlabel('Distance along Path4 (mm)', fontsize=fs*1.5)
    plt.ylabel('%Variance PC_3', fontsize=fs*1.5)
    ax = plt.gca()
    width = 3
    ax.xaxis.set_tick_params(width=width, direction='in')
    ax.yaxis.set_tick_params(width=width, direction='in')
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)  
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    plt.subplots_adjust(left=0.12)
    plt.savefig('feature_evo_CP_radial.png', dpi=300)
    plt.close('all')

    if debug:
        ngb_img = ngb.astype(np.uint8) * 128
        #ngb_img[cp_mask] = 255
        cv2.imwrite('ngb.png', ngb_img)

     

def feature_evolution_CP(mefile, debug=True):
    '''
    ['', '1', '2', '2a', '2b', '2/3', '3', '4', '4/5', '5', '5/6', '6a','6b', '6']:
    '''
    mask = load_image(MASK_CCF25_FILE)
    shape3d = mask.shape
    rids = [672]
    axid = 1
    left_axid = 0

    # estimate the mip with features
    temp_file = 'temp.pkl'
    if os.path.exists(temp_file):
        with open(temp_file, 'rb') as fp:
            mips = pickle.load(fp)
    else:
        mips = get_me_mips(mefile, shape3d, False, True, 'composite', 0)
        with open(temp_file, 'wb') as fp:
            pickle.dump(mips, fp)

    mask2d = np.take(mask, shape3d[axid]//2, axid)

    # mask for target regions
    for i, v in enumerate(rids):
        if i == 0:
            m = mask2d == v
        else:
            m = m | (mask2d == v)
    if left_axid == 0:
        m[:m.shape[0]//2] = 0
    elif left_axid == 1:
        m[:,:m.shape[1]//2] = 0
    # remove isolated points
    m_conv = cv2.filter2D(m.astype(np.uint8), ddepth=-1, kernel=np.ones((3,3),dtype=int))
    m[m_conv == 1] = 0


    # get medial axis
    skel = morphology.medial_axis(m)
    #skel = morphology.skeletonize(m, method='lee')
    # The skel may be multi-headed, we use the longest one
    skel = skel.astype(np.uint8)

    fil = FilFinder2D(skel, distance=250 * u.pc, mask=skel)
    fil.preprocess_image(flatten_percent=90)
    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    # we should extend the skeleteon until outside of the region
    ma_pts_sum = cv2.filter2D(fil.skeleton_longpath, ddepth=-1, kernel=np.ones((3,3),dtype=int))
    ep_mask = (ma_pts_sum == 2) & (fil.skeleton_longpath > 0)
    ep_pts = np.nonzero(ep_mask)

    # get the nearest points on skeleton 
    mask_pts = np.stack(np.nonzero(m)).transpose()
    ma_pts = np.stack(np.nonzero(fil.skeleton_longpath)).transpose()
    kdtree = KDTree(ma_pts, leaf_size=2)
    #dmin1, imin1 = kdtree.query(mask_pts, k=1)
    # get the rotation matrices
    #anchors = np.stack((np.zeros(ma_pts.shape[0]), np.arange(-ma_pts.shape[0]+1,1))).transpose()
    anchors = np.stack((np.arange(ma_pts.shape[0])*-0.2, np.arange(-ma_pts.shape[0]+1,1))).transpose()
    
    pt_anchor = np.array([[ep_pts[0][1], ep_pts[1][1]]])
    ma_pts_shift = ma_pts - pt_anchor
    
    v11 = (anchors * ma_pts_shift).sum(axis=1)
    v12 = np.cross(ma_pts_shift, anchors)
    rmat = np.array([v11, -v12, v12, v11]) / (np.linalg.norm(anchors, axis=1) * np.linalg.norm(ma_pts_shift, axis=1) +1e-10)
    # manually set the last point as identity
    i_idx = np.nonzero(np.fabs(ma_pts_shift).sum(axis=1) == 0)[0]
    rmat[:,i_idx[0]] = [0,-1,1,0]
    rmat = rmat.transpose().reshape((-1,2,2))

    # map all points to new space
    mip = mips[axid]
    cp_fg = (mip.sum(axis=-1) > 0) & m
    cp_fg_crds = np.stack(cp_fg.nonzero()).transpose()
    cp_fg_shift = cp_fg_crds - pt_anchor
    rmat_cp = rmat[kdtree.query(cp_fg_crds, k=1)[1][:,0]]
    cp_fg_rotated = np.einsum('BNi,Bi ->BN', rmat_cp, cp_fg_shift) + pt_anchor
    #values = mip[cp_fg].transpose().reshape(-1, 1)
    #cp_fg_rotated3 = np.vstack((cp_fg_rotated, cp_fg_rotated, cp_fg_rotated))
    #cp_features = np.hstack((cp_fg_rotated3, values))
    #df_cp = pd.DataFrame(cp_features, columns=['h', 'w', 'fvalue'])
    #df_cp['ftype'] = [__MAP_FEATS__[0] for i in range(rmat_cp.shape[0])] + \
    #                 [__MAP_FEATS__[1] for i in range(rmat_cp.shape[0])] + \
    #                 [__MAP_FEATS__[2] for i in range(rmat_cp.shape[0])]
    #sns.lmplot(data=df_cp, x='h', y='fvalue', hue='ftype')
    cp_features = np.hstack((cp_fg_rotated, mip[cp_fg]))
    df_cp = pd.DataFrame(cp_features, columns=['h', 'w', __MAP_FEATS__[0], __MAP_FEATS__[1], __MAP_FEATS__[2]])
    sns.lmplot(data=df_cp, x='h', y='pca_vr3')
    plt.savefig('temp.png', dpi=300)
    plt.close('all')
    
    
    # optional visualization
    if debug:
        ma_pts_rotated = np.einsum('BNi,Bi ->BN', rmat, ma_pts_shift)
        ma_pts_rotated = np.round(ma_pts_rotated).astype(int) + pt_anchor
        rmat_mask = rmat[imin1[:,0]]
        mask_pts_rotated = np.einsum('BNi,Bi ->BN', rmat_mask, mask_pts - pt_anchor)
        mask_pts_rotated = np.round(mask_pts_rotated).astype(int) + pt_anchor
        
        plt.scatter(mask_pts[:,1], mask_pts[:,0], color='r', alpha=.3, s=1)
        plt.scatter(ma_pts[:,1], ma_pts[:,0], color='k', s=1)
        plt.scatter(mask_pts_rotated[:,1], mask_pts_rotated[:,0], c='g', s=1, alpha=0.3)
        plt.scatter(ma_pts_rotated[:,1], ma_pts_rotated[:,0], c='k', s=1)
        # plot shift vector
        ma_pts_t = np.stack((ma_pts, ma_pts_rotated), axis=1)
        for pts in ma_pts_t:
            if np.random.random() > 0.1: continue
            plt.plot(pts[:,1], pts[:,0], color='b')
        #mask_pts_t = np.stack((mask_pts, mask_pts_rotated), axis=1)
        #for pts in mask_pts_t:
        #    if np.random.random() > 0.01: continue
        #    if np.linalg.norm(pts[0] - pts[1]) < 30: continue
        #    plt.plot(pts[:,1], pts[:,0], color='b')

        plt.title('Medial axis')
        plt.axis('off')
        plt.savefig('median_axis.png', dpi=200)
        plt.close('all')
    


if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    mapfile = 'microenviron_map'
    scale = 25.
    flip_to_left = True
    mode = 'composite'
    findex = 0
    fmt = 'png'

    '''
    # save outline
    mask = load_image(MASK_CCF25_FILE)
    for axid in range(3):
        figname = f'section_outline_axis{axid}.png'
        mip = np.take(mask, mask.shape[axid]//2, axid)
        mip[:] = 0
        process_mip(mip, mask, sectionX=None, axis=axid, figname=figname, mode='composite', with_outline=False)
    '''
    

    generate_me_maps(mefile, outfile=mapfile, flip_to_left=flip_to_left, mode=mode, findex=findex, fmt=fmt)
    #plot_left_right_corr(mefile, outfile=mapfile, histeq=True, mode='composite', findex=0)
    #colorize_atlas2d_cv2(annot=True, fmt=fmt)

    #sectional_dsmatrix(mefile, 'me_dsmatrix', histeq=False, flip_to_left=True, mode=mode, findex=findex)
    #dsfile = 'me_dsmatrix_mip1.csv'
    #dsfile_histeq = 'me_dsmatrix_mip1_histeq.csv'
    #plot_me_dsmatrix(dsfile, dsfile_histeq, axid=1)

    #feature_evolution_CP_radial(mefile, debug=False)
    
