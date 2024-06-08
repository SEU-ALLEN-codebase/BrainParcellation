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
import hdbscan
import matplotlib.lines as mlines

from swc_handler import parse_swc, write_swc, get_specific_neurite
from image_utils import get_mip_image, image_histeq
from math_utils import get_exponent_and_mantissa
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, ANATOMY_TREE_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, \
                                get_brain_mask2d, get_section_boundary, detect_edges2d
from anatomy.anatomy_core import parse_ana_tree
from global_features import calc_global_features_from_folder


# plot the top 3 features on
# features selected by mRMR
sys.path.append('../..')
from config import mRMR_f3 as __MAP_FEATS__, standardize_features
sys.path.append('../../../common_lib')
from configs import __FEAT_NAMES__


sns.set_theme(style='ticks', font_scale=1.6)


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

def plot_region_feature_sections(mefile, rname='MOB', name='auto', flipLR=True, thickX2=10, step=20, feat_names=None):
    df = pd.read_csv(mefile, index_col=0)
    if feat_names is None:
        feat_names = __MAP_FEATS__

    keys = [key for key in feat_names]
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
    memap[coords_s[:,2]-zmin, coords_s[:,1]-ymin, coords_s[:,0]-xmin] = dfc.values

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



def local_to_axon_separate(local_file, axon_file, 
                         rnames=['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']):
    dfl = pd.read_csv(local_file, index_col=0)
    dfa = pd.read_csv(axon_file, index_col=0)

    #feat_names = ['AverageContraction', 'Nodes', 'Branches', 'Tips']
    #feat_names = ['Length', 'Bifurcations', 'AverageFragmentation', 'HausdorffDimension']
    #feat_names = ['pc11', 'pc12', 'pc13']
    feat_names = ['pca_vr1', 'pca_vr2', 'pca_vr3']

    dfl = dfl[dfl.region_name.isin(rnames)][feat_names]
    dfa = dfa[dfa.region_name.isin(rnames)][feat_names]

    # remove NA values
    na_flag = (dfl.isna().sum(axis=1) + dfa.isna().sum(axis=1)) == 0
    dfl = dfl[na_flag]
    dfa = dfa[na_flag]
    print(dfl.shape, dfa.shape)

    #standardize_features(dfl, feat_names)
    #standardize_features(dfa, feat_names)

    ###### Comparison of single separate features
    df = []
    for irow, row1 in dfl.iterrows():
        row2 = dfa.loc[irow]
        for ifeat, fname in enumerate(feat_names):
            df.append([row1.iloc[ifeat], row2.iloc[ifeat], fname])

    col1, col2 = 'Dendrite', 'Axon'
    df = pd.DataFrame(df, columns=[col1, col2, 'Feature'])
    
    sns.set_theme(style="ticks", font_scale=1.9)
    g = sns.lmplot(data=df, x=col1, y=col2, col='Feature',
                   facet_kws={'sharex': False, 'sharey': False},
                   scatter_kws={'color': 'black'}, 
                   line_kws={'color': 'red'})

    # add annotate
    def annotate(data, **kws):
        gg = data['Feature'].unique()[0]

        # annotate the line fitting stats
        r, p = stats.pearsonr(data[col1], data[col2])
        ax = plt.gca()
        if gg == 'AverageContraction':
            y1, y2 = 0.16, 0.06
        else:
            y1, y2 = 0.8, 0.7

        ax.text(0.55, y1, r'$R={:.2f}$'.format(r),
                transform=ax.transAxes, color='r')
        e, m = get_exponent_and_mantissa(p)
        ax.text(0.55, y2, r'$P={%.1f}x10^{%d}$' % (m, e),
                transform=ax.transAxes, color='r')

        # Title
        if gg.startswith('Average'):
            gg = gg.replace('Average', 'Avg')
        ax.set_title(gg)
        if gg == 'AvgBifurcationAngleRemote':
            ax.set_xlabel('Local (degree)')

    g.map_dataframe(annotate)
    plt.savefig('dendrite_axon_features_hip.png', dpi=300)
    plt.close()


def clustering_on_umap(df, feat_names, nclusters=4, plot=False, figstr=''):
    df = df.copy()
    standardize_features(df, feat_names)
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(df)
    # clustering
    db = sklearn.cluster.SpectralClustering(n_clusters=nclusters).fit(embedding)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.labels_ != -1] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_:d}")
    print(f"Estimated number of noise points: {n_noise_:d}")
    # visualize
    if plot:
        # plotting
        unique_labels = set(labels)
        #colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]
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
                label = f"class={k}"
            )

            xy = embedding[class_member_mask & ~core_samples_mask]
            ax.plot(
                xy[:,0],
                xy[:,1],
                "o",
                c=tuple(col),
                markersize=2
            )
        #plt.title('Clustering of arbors')
        leg = plt.legend()
        ax.legend_.remove()
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        plt.axis('equal')
        plt.xlabel('UMAP_1')
        plt.ylabel('UMAP_2')
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.title(figstr)
        plt.savefig(f'cluster_{figstr}.png', dpi=300)
        plt.close()

    return embedding, labels
    

def local_to_axon_all(local_file, axon_file, 
                         rnames=['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']):
    dfl = pd.read_csv(local_file, index_col=0)
    dfa = pd.read_csv(axon_file, index_col=0)

    # The angles for axons have many nan values, we just remove them
    except_feats = ['AverageBifurcationAngleLocal', 'AverageBifurcationAngleRemote']
    feat_names = __FEAT_NAMES__ + ['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3']
    feat_names = [feat for feat in feat_names if feat not in except_feats]
    
    dfl = dfl[dfl.region_name.isin(rnames)][feat_names]
    dfa = dfa[dfa.region_name.isin(rnames)][feat_names]

    # remove NA values
    na_flag = (dfl.isna().sum(axis=1) + dfa.isna().sum(axis=1)) == 0
    dfl = dfl[na_flag]
    dfa = dfa[na_flag]
    print(dfl.shape, dfa.shape)

    # 
    emb_l, lab_l = clustering_on_umap(dfl, feat_names, nclusters=1, plot=True, figstr='dendrite')
    emb_a, lab_a = clustering_on_umap(dfa, feat_names, nclusters=7, plot=True, figstr='axon')
    
    print()


class AxonalProjection:
    def __init__(self):
        pass

    def calc_proj_matrix(self, axon_dir, proj_csv):
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
        projs = pd.DataFrame(np.zeros((len(axon_files), len(regids))), index=fnames)
               
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
            


if __name__ == '__main__':
    # dendrite vs axon 
    axon_dir = './ION_HIP/swc_axons_8um'
    axon_gf_file = './ION_HIP/gf_hip_axons_8um.csv'
    axon_feat_file = './ION_HIP/lm_features_d28_axons_8um.csv'
    rnames = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
    
    
    if 0:
        calc_global_features_from_folder(axon_dir, outfile=axon_gf_file)

    if 0:
        #feat_names = ['Bifurcations', 'Length', 'AverageFragmentation']
        plot_region_feature_sections(axon_feat_file, rname=rnames, name='axon', flipLR=True, thickX2=10)

    if 0:
        # plot the dendrite vs axon relationship for manually annotated CP neurons
        local_file = 'ION_HIP/lm_features_d28_dendrites.csv'
        local_to_axon_separate(local_file, axon_feat_file)
        #local_to_axon_all(local_file, axon_feat_file)
        
    if 1:
        with_local = False
        proj_csv = './ION_HIP/axon_proj_8um.csv'
        if not with_local:
            proj_csv = './ION_HIP/axon_proj_8um_withoutLocal.csv'
        ap = AxonalProjection()
        ap.calc_proj_matrix(axon_dir, proj_csv)

