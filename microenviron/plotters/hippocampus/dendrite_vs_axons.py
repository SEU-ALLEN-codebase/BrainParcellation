##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-06
#Description:               
##########################################################
import os
import glob
import re
import sys
import time
import random
import numpy as np
from collections import Counter
import numbers
import pickle
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from skimage import exposure, filters, measure
from skimage import morphology
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
from scipy import stats
import matplotlib
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import matplotlib.lines as mlines

from swc_handler import parse_swc, write_swc, get_specific_neurite
from image_utils import get_mip_image, image_histeq
from math_utils import get_exponent_and_mantissa
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, ANATOMY_TREE_FILE, \
                                   BSTRUCTS7, BSTRUCTS13, SALIENT_REGIONS
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, \
                                get_brain_mask2d, get_section_boundary, detect_edges2d
from anatomy.anatomy_core import parse_ana_tree, get_struct_from_id_path
from global_features import calc_global_features_from_folder


# plot the top 3 features on
# features selected by mRMR
sys.path.append('../..')
from config import mRMR_f3 as __MAP_FEATS__, standardize_features
sys.path.append('../../../common_lib')
from configs import __FEAT_NAMES__
from hpf_config import __RNAMES__, __RIDS__


sns.set_theme(style='ticks', font_scale=1.7)


def aggregate_meta_information(swc_dir, gf_file, out_file):
    atlas = load_image(MASK_CCF25_FILE)
    ana_tree = parse_ana_tree(keyname='id')
    shape3d = atlas.shape
    df1 = pd.read_csv(gf_file, index_col=0)
    coords = []
    regids = []
    regnames = []

    for prefix in df1.index:
        swc_file = os.path.join(swc_dir, f'{prefix}.swc')
        with open(swc_file) as fp:
            soma_str = re.search('.* -1\n', fp.read()).group()
        spos = np.array(list(map(float, soma_str.split()[2:5])))
        spos_25 = spos / 25.

        # get the brain region for each neuron
        xi,yi,zi = np.floor(spos_25).astype(int)
        if (xi >= shape3d[2]) or (yi >= shape3d[1]) or (zi >= shape3d[0]):
            regid = 0
        else:
            regid = atlas[zi,yi,xi]
        
        if regid != 0:
            regname = ana_tree[regid]['acronym']
        else:
            regname = ''
        
        coords.append(np.round(spos, 3))
        regids.append(regid)
        regnames.append(regname)

        if len(regids) % 20 == 0:
            print(len(regids))

    coords = np.array(coords)
    df1['soma_x'] = coords[:,0]
    df1['soma_y'] = coords[:,1]
    df1['soma_z'] = coords[:,2]
    df1['region_id_r671'] = regids
    df1['region_name_r671'] = regnames

    df_cp = df1[df1['region_id_r671'] == 672]
    df_cp.to_csv(out_file)



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

def plot_region_feature_sections(mefile, rname='MOB', name='auto', flipLR=True, thickX2=10, step=20, feat_names=None, feat_type='me', mode='composite', findex=0):
    df = pd.read_csv(mefile, index_col=0)
    if feat_names is None:
        feat_names = __MAP_FEATS__

    if feat_type == 'me':
        keys = [f'{key}_me' for key in feat_names]
    elif feat_type == 'single':
        keys = feat_names
    elif feat_type == 'global_pca':
        keys = ['pca_feat1', 'pca_feat2', 'pca_feat3']
    elif feat_type == 'local_me_pca':
        keys = [key for key in df.columns if key.endswith('_me')]
    elif feat_type == 'local_single_pca':
        keys = [key[:-3] for key in df.columns if key.endswith('_me')]
        if len(keys) == 0:
            from config import __FEAT24D__
            keys = __FEAT24D__
    else:
        raise ValueError

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
    # remove null samples
    sel_mask = sel_mask & (df[keys].isna().sum(axis=1) == 0)

    dfr = df[keys][sel_mask]
    if (feat_type == 'local_me_pca') or (feat_type == 'local_single_pca'):
        # do pca feature reduction
        pca = PCA(n_components=3, whiten=True)
        dfr = pd.DataFrame(pca.fit_transform(dfr), columns=('pca_feat1', 'pca_feat2', 'pca_feat3'))
    
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
    nzcoords = rmask.nonzero()
    nzcoords_t = np.array(nzcoords).transpose()
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)
    sub_mask = rmask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    memap = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)

    coords_s = np.floor(coords * 40).astype(int)

    if mode == 'composite':
        memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = dfc.values
    elif mode == 'single':
        memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = dfc[:,findex].reshape(-1,1)
    else:
         memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = [255,0,0]

    mips = []
    shape3d = mask.shape
    axid = 2
    for sid in range(0, xmax-xmin-thickX2-1, step):
        sid = sid + step//2
        cur_memap = memap.copy()
        cur_memap[:,:,:sid-thickX2] = 0
        cur_memap[:,:,sid+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())

        mip = get_mip_image(cur_memap, axid)

        figname = f'{out_prefix}_section{sid:03d}_{name}_axon.png'
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


def clustering_on_umap(df, feat_names=None, nclusters=4, plot=False, figstr='', precomputed_labels=None, seed=1024):
    df = df.copy()
    if feat_names is not None:
        standardize_features(df, feat_names)
    
    if df.shape[1] != 2:
        reducer = umap.UMAP(random_state=seed)
        embedding = reducer.fit_transform(df)
    else:
        embedding = df

    if precomputed_labels is None:
        # clustering
        db = sklearn.cluster.SpectralClustering(n_clusters=nclusters, random_state=seed, n_jobs=8).fit(embedding)
        # I would like to sort the labels, so that their colors will not change run-by-run
        labels = db.labels_
        sorted_labels = np.zeros_like(labels)
        unique_labels = np.unique(labels)
        # sorting criterion
        means = [embedding[labels == label].mean(axis=0) for label in unique_labels]
        random.seed(seed)
        random.shuffle(means)
        sorted_indices = np.argsort([mean[0] for mean in means])
        # map the original labels to sorted labels
        for new_label, old_label in enumerate(sorted_indices):
            sorted_labels[labels == unique_labels[old_label]] = new_label
        labels = sorted_labels

    else:
        labels = precomputed_labels
        # estimate the relative
        if np.unique(labels).shape[0] != 1:
            print(silhouette_score(embedding, labels))
            print(davies_bouldin_score(embedding, labels))
            print(calinski_harabasz_score(embedding, labels))
    
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[labels != -1] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_:d}")
    print(f"Estimated number of noise points: {n_noise_:d}")
    # visualize
    # plotting
    unique_labels = sorted(set(labels))
    #colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]
    if plot:
        # map the features to 2D for better visualization
        fig, ax = plt.subplots(figsize=(6,6))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # black for noise
                col = [0, 0, 0, 1]

            class_member_mask = labels == k
            print(f'==> Class {k} has #samples={class_member_mask.sum()}')

            xy = embedding[class_member_mask & core_samples_mask]
            ax.plot(
                xy[:,0],
                xy[:,1],
                "o",
                c=tuple(col),
                markersize=2,
                alpha = 0.75,
                label = f"cluster{k}"
            )

            xy = embedding[class_member_mask & ~core_samples_mask]
            ax.plot(
                xy[:,0],
                xy[:,1],
                "o",
                c=tuple(col),
                markersize=2,
                alpha = 0.75
            )
        #plt.title('Clustering of arbors')
        ax_leg = ax.legend(labelspacing=0.0, handletextpad=0.,
                   borderpad=0.05, frameon=False, loc='upper right',
                   fontsize=15, alignment='center', ncols=3,
                   markerscale=4, columnspacing=0.5)
        ax_leg._legend_box.align = 'center'
        #ax.legend_.remove()
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_visible(False) #.set_linewidth(2)
        ax.spines['top'].set_visible(False) #.set_linewidth(2)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # customize the location of spines
        #xmin, ymin = embedding.min(axis=0)
        #xmax, ymax = embedding.max(axis=0)
        #ax.spines['left'].set_position(("data", xmin))
        #ax.spines['bottom'].set_position(("data", ymin))
        
        #ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        #ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        
        #ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('equal')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.title(figstr)
        plt.savefig(f'cluster_{figstr}.png', dpi=300)
        plt.close()

    return embedding, labels, colors

def plot_labels_on_ccf(soma_xyzs, labels, colors, flipLR=True, thickX2=10, step=20, out_prefix='tmp'):
    mask = load_image(MASK_CCF25_FILE)
    ana_tree = parse_ana_tree(keyname='name')

    # regional mask
    rmask = np.zeros_like(mask)
    for ri in __RNAMES__:
        idx = ana_tree[ri]['id']
        rmask = rmask | (mask == idx)

    coords = soma_xyzs.values / 1000
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
    colors = [np.floor(np.array(color[:3]) * 255.).astype(np.uint8) for color in colors]
    mycolors = [colors[label] for label in labels]


    # get the boundary of region
    nzcoords = rmask.nonzero()
    nzcoords_t = np.array(nzcoords).transpose()
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)
    sub_mask = rmask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    memap = np.zeros((*sub_mask.shape, 3), dtype=np.uint8)

    coords_s = np.floor(coords * 40).astype(int)

    memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = mycolors

    mips = []
    shape3d = mask.shape
    axid = 2
    for sid in range(0, xmax-xmin-thickX2-1, step):
        sid = sid + step//2
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



def local_axon_emb_clustering(local_file, axon_file, is_local_me=False, plot=True):

    if type(local_file) is str:
        dfl_o = pd.read_csv(local_file, index_col=0)
    else:
        dfl_o = local_file

    if type(axon_file) is str:
        dfa_o = pd.read_csv(axon_file, index_col=0)
    else:
        dfa_o = axon_file

    # The angles for axons have many nan values, we just remove them
    except_feats = ['AverageBifurcationAngleLocal', 'AverageBifurcationAngleRemote']
    feat_names = __FEAT_NAMES__ + ['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3']
    feat_names = [feat for feat in feat_names if feat not in except_feats]
    if is_local_me:
        feat_names_local = [f'{name}_me' for name in feat_names]
    else:
        feat_names_local = feat_names
    
    dfl = dfl_o[dfl_o.region_name.isin(__RNAMES__)][feat_names_local]
    dfa = dfa_o[dfa_o.region_name.isin(__RNAMES__)][feat_names]

    # remove NA values
    na_flag = (dfl.isna().sum(axis=1) + dfa.isna().sum(axis=1)) == 0
    dfl = dfl[na_flag]
    dfa = dfa[na_flag]
    print(dfl.shape, dfa.shape)

    # 
    if is_local_me:
        local_figstr = 'Microenvironment'
        nclusters = 3
    else:
        local_figstr = 'Dendrite'
        nclusters = 1
    emb_l, lab_l, colors_l = clustering_on_umap(dfl, feat_names_local, nclusters=nclusters, plot=plot, figstr=local_figstr)
    # coloring according labels
    plot_labels_on_ccf(dfl_o.loc[dfl.index][['soma_x', 'soma_y', 'soma_z']], lab_l, colors_l, flipLR=True, thickX2=10, step=20, out_prefix=f'cluster_labels_{local_figstr}')

    emb_a, lab_a, colors_a = clustering_on_umap(dfa, feat_names, nclusters=7, plot=plot, figstr='Axon')
    plot_labels_on_ccf(dfa_o.loc[dfa.index][['soma_x', 'soma_y', 'soma_z']], lab_a, colors_a, flipLR=True, thickX2=10, step=20, out_prefix=f'cluster_labels_axon')
    
    return emb_l, lab_l, emb_a, lab_a, na_flag

def dendrite_axon_correspondence(local_file, axon_file, is_local_me=False):
    emb_l, lab_l, emb_a, lab_a, _ = local_axon_emb_clustering(local_file, axon_file, is_local_me, plot=False)
    if is_local_me:
        local_figstr = 'me_by_axon'
        axon_figstr = 'axon_by_me'
    else:
        local_figstr = 'dendrite_by_axon'
        axon_figstr = 'axon_by_dendrite'
    clustering_on_umap(emb_l, feat_names=None, nclusters=4, plot=True,
                           figstr=local_figstr, precomputed_labels=lab_a)
    clustering_on_umap(emb_a, feat_names=None, nclusters=4, plot=True,
                           figstr=axon_figstr, precomputed_labels=lab_l)
    # visualize using sankey plot
    


class AxonalProjection:
    def __init__(self):
        pass

    def calc_proj_matrix(self, axon_dir, proj_csv, reg_csv='./ION_HIP/lm_features_d28_axons_8um.csv'):
        if os.path.exists(proj_csv):
            projs = pd.read_csv(proj_csv, index_col=0)
        else:
            atlas = load_image(MASK_CCF25_FILE)
            zdim, ydim, xdim = atlas.shape
            # get the new atlas with differentiation of left-right hemisphere
            atlas_lr = np.zeros(atlas.shape, dtype=np.int64)
            atlas_lr[:zdim//2] = atlas[:zdim//2]
            atlas_lr[zdim//2:] = -atlas[zdim//2:].astype(np.int64)
            # vector
            regids = np.unique(atlas_lr[atlas_lr != 0])
            rdict = dict(zip(regids, range(len(regids))))

            axon_files = glob.glob(os.path.join(axon_dir, '*swc'))
            fnames = [os.path.split(fname)[-1][:-4] for fname in axon_files]
            projs = pd.DataFrame(np.zeros((len(axon_files), len(regids))), index=fnames, columns=regids)
                   
            t0 = time.time()
            for iaxon, axon_file in enumerate(axon_files):
                ncoords = pd.read_csv(axon_file, sep=' ', usecols=(2,3,4,6)).values
                # flipping
                smask = ncoords[:,-1] == -1
                # convert to CCF-25um
                ncoords[:,:-1] = ncoords[:,:-1] / 25.
                soma_coord = ncoords[smask][0,:-1]
                ncoords = ncoords[~smask][:,:-1]
                if soma_coord[2] > zdim/2:
                    ncoords[:,2] = zdim - ncoords[:,2]
                # make sure no out-of-mask points
                ncoords = np.round(ncoords).astype(int)
                ncoords[:,0] = np.clip(ncoords[:,0], 0, xdim-1)
                ncoords[:,1] = np.clip(ncoords[:,1], 0, ydim-1)
                ncoords[:,2] = np.clip(ncoords[:,2], 0, zdim-1)
                # get the projected regions
                proj = atlas_lr[ncoords[:,2], ncoords[:,1], ncoords[:,0]]
                # to project matrix
                rids, rcnts = np.unique(proj, return_counts=True)
                # Occasionally, there are some nodes located outside of the atlas, due to 
                # the registration error
                nzm = rids != 0
                rids = rids[nzm]
                rcnts = rcnts[nzm]
                rindices = np.array([rdict[rid] for rid in rids])
                projs.iloc[iaxon, rindices] = rcnts

                if (iaxon + 1) % 10 == 0:
                    print(f'--> finished {iaxon+1} in {time.time()-t0:.2f} seconds')

            projs *= 8 # to um scale
            projs.to_csv(proj_csv)

        # zeroing non-salient regions
        df_reg = pd.read_csv(reg_csv, index_col=0)
        target_neurons = [rname for rname in df_reg.index[df_reg.region_name.isin(__RNAMES__)]]
        projs = projs.loc[target_neurons]

        salient_mask = np.array([True if np.fabs(int(col)) in SALIENT_REGIONS else False for col in projs.columns])
        keep_mask = (projs.sum() > 0) & salient_mask
        # filter the neurons not in target regions
        projs = projs.loc[:, keep_mask]

        # plot the map
        ana_tree = parse_ana_tree()
        # regid to regname
        regnames = []
        bstructs = []
        for regid in projs.columns:
            regid = int(regid)
            if regid > 0:
                hstr = 'ipsi'
            else:
                hstr = 'contra'
            regid = np.fabs(regid)
            regname = hstr + ana_tree[regid]['acronym']
            regnames.append(regname)
            
            bstruct = get_struct_from_id_path(ana_tree[regid]['structure_id_path'], BSTRUCTS7)
            if bstruct == 0:
                print(regname, regid)
                bstructs.append('root')
            else:
                bstructs.append(ana_tree[bstruct]['acronym'])
        

    def clustering(self, proj_csv, local_file, axon_file, 
                   to_log_space=True, is_local_me=False):
        projs = pd.read_csv(proj_csv, index_col=0)
        local = pd.read_csv(local_file, index_col=0) # reference of the meta information
        axons = pd.read_csv(axon_file, index_col=0)

        emb_l, lab_l, emb_a, lab_a, na_flag = local_axon_emb_clustering(local, axons, is_local_me, plot=True)
        
        #make sure the share the same index order 
        projs = projs[axons.region_name.isin(__RNAMES__)][na_flag]
        # keep only the salient 
        projs.mask(projs < 1000, 0, inplace=True)
        # remove zero-length regions
        projs.drop(projs.columns[projs.sum(axis=0) == 0], axis=1, inplace=True)
        
        if to_log_space:
            # to log-space
            projs = np.log(projs + 1)
        
        if is_local_me:
            local_figstr = 'Microenvironment'
        else:
            local_figstr = 'Dendrite'

        emb, lab, colors = clustering_on_umap(projs, feat_names=None, nclusters=9, plot=True, 
                           figstr='Projection')
        plot_labels_on_ccf(local.loc[projs.index][['soma_x', 'soma_y', 'soma_z']], 
                           lab, colors, flipLR=True, thickX2=10, step=20, out_prefix=f'cluster_labels_projs')
 
        # now do the real data processing
        clustering_on_umap(projs, feat_names=None, nclusters=4, plot=True, 
                           figstr=f'Projection_by_{local_figstr}', precomputed_labels=lab_l)

        # coloring by the region
        #regions = axons.region_name[axons.region_name.isin(__RNAMES__)][na_flag].values
        #clustering_on_umap(projs, feat_names=None, nclusters=4, plot=True, 
        #                   figstr='Projection_by_regions', precomputed_labels=regions)
        
        # sankey correspondence
        labels_me = [f'c{i}_ME' for i in np.unique(lab_l)]
        labels_proj = [f'c{i}_proj' for i in np.unique(lab)]
        labels = labels_me + labels_proj
        # node colors
        #lut = dict(zip(np.unique(labels), sns.hls_palette(len(np.unique(labels)), l=0.5, s=0.8)))
        lut_me = {lab:plt.cm.rainbow(each)[:3] for lab, each in zip(labels_me, np.linspace(0, 1, len(labels_me)))}
        lut_proj = {lab:plt.cm.rainbow(each)[:3] for lab, each in zip(labels_proj, np.linspace(0, 1, len(labels_proj)))}
        lut = lut_me | lut_proj
        
        node_color_vs = pd.Series(labels, name='label').map(lut).values
        #np.random.shuffle(node_color_vs)
        node_colors = []
        for color in node_color_vs:
            r,g,b = color
            r = int(255* r)
            g = int(255 * g)
            b = int(255 * b)
            node_colors.append(f'rgb({r},{g},{b})')
        
        import plotly.graph_objects as go
        pairs = np.vstack((lab_l, lab+len(np.unique(lab_l)))).transpose()
        pindices, pcounts = np.unique(pairs, axis=0, return_counts=True)
        sources = pindices[:,0]
        targets = pindices[:,1]
        values = pcounts

        # Customize the link color
        link_colors = []
        for target in targets:
            rgb = node_colors[target]
            rgba = 'rgba' + rgb[3:-1] + f',{160})'
            link_colors.append(rgba)

        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad =15,
                thickness = 25,
                line = dict(color = "black", width = 0.5),
                label = ['' for i in range(len(labels))], #labels,
                color = node_colors
                ),
            link = dict(
                source = sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
                target = targets,
                value = values,
                color = link_colors
        ))])

        fig.update_layout(title_text="", font_size=16, width=800, height=600)
        fig.write_image(f'sankey_{local_figstr}_vs_proj.png', scale=2)
            


if __name__ == '__main__':
    # dendrite vs axon 
    axon_dir = './ION_HIP/swc_axons_8um'
    local_file = 'ION_HIP/lm_features_d28_dendrites.csv'
    local_me_file = './ION_HIP/mefeatures_dendrites.csv'
    axon_gf_file = './ION_HIP/gf_hip_axons_8um.csv'
    axon_feat_file = './ION_HIP/lm_features_d28_axons_8um.csv'
    
    
    if 0:
        for n in range(5, 40+1, 5):
            axon_dir = f'./ION_HIP/point_perturbation2/swc_dendrites_del_max{n}'
            axon_gf_file = f'./ION_HIP/point_perturbation2/gf_hip_dendrites_del_max{n}.csv'
            calc_global_features_from_folder(axon_dir, outfile=axon_gf_file, robust=True)

    if 0:
        #feat_names = ['Bifurcations', 'Length', 'AverageFragmentation']
        plot_region_feature_sections(axon_feat_file, rname=__RNAMES__, name='axon', flipLR=True, thickX2=10, mode='soma', feat_type='single')

    #if 0:
        # plot the dendrite vs axon relationship for manually annotated CP neurons
        #local_axon_emb_clustering(local_me_file, axon_feat_file, is_local_me=True)
        #dendrite_axon_correspondence(local_me_file, axon_feat_file, is_local_me=False)
        
        
    if 1:
        proj_csv = './ION_HIP/axon_proj_8um.csv'
        ap = AxonalProjection()
        #ap.calc_proj_matrix(axon_dir, proj_csv)
        ap.clustering(proj_csv, local_me_file, axon_feat_file, is_local_me=True)


