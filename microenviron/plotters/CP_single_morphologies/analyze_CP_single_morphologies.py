##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-06
#Description:               
##########################################################
import os
import glob
import re
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
from scipy import stats
import matplotlib
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from swc_handler import parse_swc, write_swc, get_specific_neurite
from image_utils import get_mip_image, image_histeq
from math_utils import get_exponent_and_mantissa
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE, ANATOMY_TREE_FILE
from anatomy.anatomy_vis import get_brain_outline2d, get_section_boundary_with_outline, \
                                get_brain_mask2d, get_section_boundary, detect_edges2d
from anatomy.anatomy_core import parse_ana_tree
from global_features import calc_global_features_from_folder, __FEAT_NAMES22__


# plot the top 3 features on
# features selected by mRMR
sys.path.append('../..')
from config import mRMR_f3 as __MAP_FEATS__

def standardize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    dfc[feat_names] = fvalues.values

def normalize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.min()) / (fvalues.max() - fvalues.min() + epsilon)
    dfc[feat_names] = fvalues.values

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

def plot_region_feature_sections(mefile, rname='MOB', name='auto', r316=False, flipLR=True, thickX2=10, step=20, feat_names=None):
    df = pd.read_csv(mefile, index_col=0)
    if feat_names is None:
        feat_names = __MAP_FEATS__

    keys = [key for key in feat_names]
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
    #print(dfr.shape); sys.exit()
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
    for sid in range(0, xmax-xmin-thickX2-1, step):
        sid = sid + step//2
        cur_memap = memap.copy()
        cur_memap[:,:,:sid-thickX2] = 0
        cur_memap[:,:,sid+thickX2:] = 0
        print(cur_memap.mean(), cur_memap.std())

        mip = get_mip_image(cur_memap, axid)

        figname = f'{out_prefix}_section{sid:03d}_{name}.png'
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


def estimate_similarity(parc_file, gf_file, is_axon=True):
    zdim = 456
        
    parc = load_image(parc_file)
    df = pd.read_csv(gf_file, index_col=0)

    coords = np.floor(df[['soma_z', 'soma_y', 'soma_x']].values / 25.).astype(int)
    regions = parc[coords[:,0], coords[:,1], coords[:,2]]

    coords_l = coords.copy()
    right_m = coords_l[:,0] < zdim/2
    right_ids = np.nonzero(right_m)[0]
    coords_l[right_ids, 0] = 456 - coords_l[right_ids, 0]
    regions = parc[coords_l[:,0], coords_l[:,1], coords_l[:,2]]
    
    nzi = np.nonzero(regions == 0)  # out-of-boundary caused by neumeric reason
    coords_l[nzi, 0] -= 1
    regions = parc[coords_l[:,0], coords_l[:,1], coords_l[:,2]]
    df['region'] = regions

    feat_names = __FEAT_NAMES22__
    
    dfc = df[feat_names + ['region']]
    standardize_features(dfc, feat_names)

    # using the region as index
    dfc.set_index('region', inplace=True)
    rs, cs = np.unique(dfc.index, return_counts=True)
    print(rs, cs)
    dfcv = dfc / np.linalg.norm(dfc, axis=1).reshape(-1,1)
    corr = pd.DataFrame(np.dot(dfcv, dfcv.transpose()), index=dfc.index, columns=dfc.index)

    reg_corrs = np.zeros((len(rs), len(rs)))
    for ir, ri in enumerate(rs):
        ids1 = np.nonzero(dfc.index == ri)[0]
        for jr in range(ir, len(rs)):
            rj = rs[jr]
            ids2 = np.nonzero(dfc.index == rj)[0]
            cur_corr = corr.iloc[ids1, ids2]
            if ir == jr:
                k = 1
            else:
                k = 0
            i_trius = np.triu_indices_from(cur_corr, k=k)
            vs = cur_corr.values[i_trius[0], i_trius[1]]
            #sns.violinplot(data=vs)
            #plt.ylim(-1, 1); plt.savefig(f'{ir}_{jr}.png'); plt.close()
            vsm = vs.mean()
            reg_corrs[ir, jr] = vsm
            reg_corrs[jr, ir] = vsm
        
    reg_corrs = pd.DataFrame(np.array(reg_corrs))
    i_trius = np.triu_indices_from(reg_corrs)
    trius = reg_corrs.values[i_trius[0], i_trius[1]]
    
    cols = np.array([(i, j) for i in range(len(rs)) for j in range(len(rs))])
    reg_values = reg_corrs.values.reshape(-1)
    #reg_values[reg_values < 0] = 0
    df = pd.DataFrame(reg_values, columns=['Similarity'])
    df['Parc1'] = cols[:,0]
    df['Parc2'] = cols[:,1]
    # to make sure the unique size legend for local and axon
    in_mask1 = (df['Similarity'] < 0)
    in_mask2 = (df['Similarity'] > 0.8)
    df.loc[np.nonzero(in_mask1)[0], 'Similarity'] = 0
    df.loc[np.nonzero(in_mask2)[0], 'Similarity'] = 0.8
    sns.set_theme(style="ticks", font_scale=1.6)
    g = sns.relplot(df, x='Parc1', y='Parc2', hue='Similarity', 
                size='Similarity', palette="afmhot_r", edgecolor="1.",
                sizes=(0, 200), size_norm=(0, 0.7), hue_norm=(0, 0.6))
    g.set(xlabel="Sub-region", ylabel="Sub-region", aspect="equal")
    g.set(xticks=list(range(len(rs))), yticks=list(range(len(rs))))
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)

    # configuring the legend
    
    
    if is_axon:
        figname = 'parc_vs_axon.png'
        plt.title('Axon')
    else:
        figname = 'parc_vs_dendrite.png'
        plt.title('Dendrite')
    plt.savefig(figname, dpi=300); plt.close()

    print(np.diagonal(reg_corrs).mean(), trius.mean())
    print(reg_corrs)


def local_to_axon_manual(local_file, axon_file):
    dfl = pd.read_csv(local_file, index_col=0)
    dfa = pd.read_csv(axon_file, index_col=0)
    feat_names = ['AverageContraction', 'AverageBifurcationAngleRemote', 
                  'HausdorffDimension', 'Bifurcations']

    dfl = dfl[feat_names]
    dfa = dfa[feat_names]
    #standardize_features(dfl, feat_names)
    #standardize_features(dfa, feat_names)

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
        ax.text(0.55, 0.16, r'$R={:.2f}$'.format(r),
                transform=ax.transAxes)
        e, m = get_exponent_and_mantissa(p)
        ax.text(0.55, 0.06, r'$P={%.1f}x10^{%d}$' % (m, e),
                transform=ax.transAxes)

        # Title
        if gg.startswith('Average'):
            gg = gg.replace('Average', 'Avg')
        ax.set_title(gg)
        if gg == 'AvgBifurcationAngleRemote':
            ax.set_xlabel('Local (degree)')

    g.map_dataframe(annotate)
    plt.savefig('dendrite_axon_features_cp.png', dpi=300)
    plt.close()


def comp_parc_and_ptype(parc_file, meta_file):
    np.random.seed(1024)
    random.seed(1024)

    parc = load_image(parc_file)
    meta = pd.read_excel(meta_file)
    # keep only the manual annotated CP neurons
    cp_neurons = meta[meta['Projection class'].isin(['CP_SNr', 'CP_GPe', 'CP_others'])]
    # get the parcellations
    coords = cp_neurons[['Soma_Z(CCFv3_1𝜇𝑚)', 'Soma_Y(CCFv3_1𝜇𝑚)', 'Soma_X(CCFv3_1𝜇𝑚)']] / 25.  # 25um
    # in parcellation
    zyx = np.floor(coords).astype(int).values
    # mirroring to left
    zdim = 456
    r_nz = np.nonzero(zyx[:,0] <= zdim/2)
    zyx[r_nz,0] = zdim - zyx[r_nz,0]
    # get the parcellations
    in_indices = np.nonzero(parc[zyx[:,0], zyx[:,1], zyx[:,2]] > 0)[0]
    # re-select the neurons
    in_zyx = zyx[in_indices]
    cp_parc = parc[in_zyx[:,0], in_zyx[:,1], in_zyx[:,2]] - 1 # start from 0
    ptypes = cp_neurons.iloc[in_indices]['Projection class']
    # sankey plot
    labels = [f'R{i}' for i in range(16)] + ['CP_GPe', 'CP_SNr', 'CP_others']
    # node colors
    lut = dict(zip(np.unique(labels), sns.hls_palette(len(np.unique(labels)), l=0.5, s=0.8)))
    node_color_vs = pd.Series(labels, name='label').map(lut).values
    np.random.shuffle(node_color_vs)
    node_colors = []
    for color in node_color_vs:
        r,g,b = color
        r = int(255* r)
        g = int(255 * g)
        b = int(255 * b)
        node_colors.append(f'rgb({r},{g},{b})')
    
    lmap = {'CP_GPe': len(labels)-3, 'CP_SNr': len(labels)-2, 'CP_others': len(labels)-1}
    ptypes_np = ptypes.map(lmap).values

    import plotly.graph_objects as go
    pairs = np.vstack((cp_parc, ptypes_np)).transpose()
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
            label = labels,
            color = node_colors
            ),
        link = dict(
            source = sources, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = targets,
            value = values,
            color = link_colors
    ))])

    fig.update_layout(title_text="", font_size=16, width=500, height=500)
    fig.write_image('parc_vs_ptypes.png', scale=2)

    print()





#----------- Section I ------------#
# get the global features of local morphologies
if 0:
    # get the soma position of SEU-A1876 from swc
    swc_dir = '../../../evaluation/data/1891_100um_2um_dendrite/'
    gf_file = '../../../evaluation/data/gf_1876_crop_2um_dendrite.csv'
    out_file = 'cp_1876_dendrite_features.csv'

    aggregate_meta_information(swc_dir, gf_file, out_file)
    
#-------------- Section II --------------#
if 0:
    # plotting
    using_auto = False
    if using_auto:
        ffile = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
        name = 'auto'
        thickX2 = 10
    else:
        ffile = 'cp_1876_dendrite_features.csv'
        name = 'dendrite'
        thickX2 = 10
    
    plot_region_feature_sections(ffile, rname='CP', name=name, r316=False, flipLR=True, thickX2=thickX2, step=20)


#--------------- Section III ---------------#
# dendrite vs axon 
if 1:
    swc_dir = '../../data/S3_1um_final/'
    axon_dir = 'cp_axons'
    axon_gf_file = 'cp_axon_gf_1876.csv'
    axon_feat_file = 'cp_1876_axonal_features.csv'
    
    
    
    if 0:   # extract all axons (along with soma)
        swc_dir = '../../data/S3_1um_final/'
        cp_file = 'cp_1876_local_features.csv'
        # extract the axons from the neurons
        df_cp = pd.read_csv(cp_file, index_col=0)
        for swc_name in df_cp.index:
            print(f'--> Processing {swc_name}')
            swc_file = os.path.join(swc_dir, f'{swc_name}.swc')
            tree = parse_swc(swc_file)
            axons = get_specific_neurite(tree, [1,2])
            out_file = os.path.join(axon_dir, f'{swc_name}.swc')
            write_swc(axons, out_file)

    if 0:
        calc_global_features_from_folder(axon_dir, outfile=axon_gf_file)
        #aggregate_meta_information(swc_dir, axon_gf_file, axon_feat_file)

    if 0:
        ffile = 'cp_1876_axonal_features.csv'
        #feat_names = ['Bifurcations', 'Length', 'AverageFragmentation']
        plot_region_feature_sections(ffile, rname='CP', name='axon', r316=False, flipLR=True, thickX2=10)

    if 0:
        # quantitative analyses
        parc_file = '../../output_full_r671/parc_region672.nrrd'
        
        for is_axon in [True, False]:
            if is_axon:
                gf_file = 'cp_1876_axonal_features.csv'
            else:
                gf_file = 'cp_1876_dendrite_features.csv'
            
            estimate_similarity(parc_file, gf_file, is_axon=is_axon)
        
        
    if 0:
        # plot the dendrite vs axon relationship for manually annotated CP neurons
        local_file = 'cp_1876_dendrite_features.csv'
        axon_file = 'cp_1876_axonal_features.csv'
        local_to_axon_manual(local_file, axon_file)

# ---------------- Section IV --------------#
# compare with existing neuron types
if 1:
    parc_file = '../../output_full_r671/parc_region672.nrrd'
    meta_file = 'TableS6_Full_morphometry_1222.xlsx'
    comp_parc_and_ptype(parc_file, meta_file)

