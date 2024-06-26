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
from sklearn import decomposition

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
from config import __FEAT24D__
from generate_me_map import plot_region_feature_sections

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
    pca_feats = []

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

        # estimate the pca features
        ccs = np.genfromtxt(swc_file, usecols=(2,3,4))
        pca = decomposition.PCA()
        pca.fit(ccs)
        pca_feats.append((*pca.components_[0], *pca.explained_variance_ratio_))

        if len(regids) % 20 == 0:
            print(len(regids))

    coords = np.array(coords)
    pca_feats = np.array(pca_feats)

    df1['soma_x'] = coords[:,0]
    df1['soma_y'] = coords[:,1]
    df1['soma_z'] = coords[:,2]
    df1['region_id_r671'] = regids
    df1['region_name_r671'] = regnames
    df1['pc11'] = pca_feats[:,0]
    df1['pc12'] = pca_feats[:,1]
    df1['pc13'] = pca_feats[:,2]
    df1['pca_vr1'] = pca_feats[:,3]
    df1['pca_vr2'] = pca_feats[:,4]
    df1['pca_vr3'] = pca_feats[:,5]

    df_cp = df1[df1['region_id_r671'] == 672]
    df_cp.to_csv(out_file)


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

    feat_names = __FEAT24D__
    
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
    plot_data = 'dendrite'
    if plot_data == 'me':
        ffile = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
        name = 'auto'
    elif plot_data == 'dendrite':
        ffile = 'cp_1876_dendrite_features.csv'
        name = 'dendrite'
    elif plot_data == 'axon':
        ffile = 'cp_1876_axonal_features.csv'
        name = 'axon'
    
    plot_region_feature_sections(ffile, rname='CP', feat_type='local_single_pca')


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
        #calc_global_features_from_folder(axon_dir, outfile=axon_gf_file)
        aggregate_meta_information(swc_dir, axon_gf_file, axon_feat_file)

    if 1:
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
if 0:
    parc_file = '../../output_full_r671/parc_region672.nrrd'
    meta_file = 'TableS6_Full_morphometry_1222.xlsx'
    comp_parc_and_ptype(parc_file, meta_file)

