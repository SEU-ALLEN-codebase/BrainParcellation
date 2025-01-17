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
import umap
from skimage import exposure, filters, measure
from skimage import morphology
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
from scipy import stats
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
import matplotlib
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sklearn
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
sys.path.append('../')
from config import mRMR_f3 as __MAP_FEATS__
from config import __FEAT24D__
from generate_me_map import plot_region_feature_sections, process_mip

def standardize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    dfc[feat_names] = fvalues.values

def clustering_on_umap(df, feat_names=None, nclusters=4, plot=False, figstr='', precomputed_labels=None, seed=1024):
    df = df.copy()
    if feat_names is not None:
        standardize_features(df, feat_names)

    if df.shape[1] != 2:
        reducer = umap.UMAP(random_state=seed)
        embedding = reducer.fit_transform(df[feat_names])
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
    colors = [plt.cm.rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]
    if figstr == 'Projection':
        colors = _PROJ_COLORS
    elif figstr == ('Microenvironment') or (figstr == 'Projection_by_Microenvironment'):
        colors = _ME_COLORS

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
                markersize=6,
                alpha = 0.75,
                label = f"cluster{k+1}"
            )

            xy = embedding[class_member_mask & ~core_samples_mask]
            ax.plot(
                xy[:,0],
                xy[:,1],
                "o",
                c=tuple(col),
                markersize=6,
                alpha = 0.75
            )
        #plt.title('Clustering of arbors')
        ax_leg = ax.legend(labelspacing=0.2, handletextpad=0.1,
                   borderpad=0.1, frameon=False, loc='upper right',
                   fontsize=15, alignment='center', ncols=1,
                   markerscale=1.8, columnspacing=0.5)
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

def compare_features(df1, df2, figname):
    sns.set_theme(style='ticks', font_scale=2)

    pf2label = {
        #'AverageBifurcationAngleRemote': 'Bif angle remote',
        #'AverageBifurcationAngleLocal': 'Bif angle local',
        'AverageContraction': 'Contraction',
        'AverageFragmentation': 'Avg. Fragmentation',
        #'AverageParent-daughterRatio': 'Avg. PD ratio',
        #'Bifurcations': 'No. of bifs',
        'Branches': 'No. of branches',
        'HausdorffDimension': 'Hausdorff dimension',
        'MaxBranchOrder': 'Max. branch order',
        'Length': 'Total length',
        #'MaxEuclideanDistance': 'Max. Euc distance',
        #'MaxPathDistance': 'Max. path distance',
        #'Nodes': 'No. of nodes',
        #'OverallDepth': 'Overall z span',
        #'OverallHeight': 'Overall y span',
        #'OverallWidth': 'Overall x span',
        #'Tips': 'No. of tips',
    }

    df1_melted = df1[pf2label.keys()].melt(var_name='Feature', value_name='Value')
    df1_melted['Dataset'] = 'R1'

    df2_melted = df2[pf2label.keys()].melt(var_name='Feature', value_name='Value')
    df2_melted['Dataset'] = 'R13'

    combined_df = pd.concat([df1_melted, df2_melted], ignore_index=True)

    fig = plt.figure(figsize=(10, 10))
    sns.boxplot(x='Feature', y='Value', hue='Dataset', data=combined_df,
                width=0.45, 
                boxprops=dict(linewidth=2),       # boundary of the box
                whiskerprops=dict(linewidth=2),   # whiskers
                capprops=dict(linewidth=2),       # caps at the ends of whiskers
                medianprops=dict(linewidth=2)     # median line
                )

    plt.xticks(rotation=45, rotation_mode='anchor', ha='right', va='top')
    plt.xlabel('Morphological feature')
    plt.ylabel('Standardized feature value')

    ax = plt.gca()
    ax_leg = ax.legend(labelspacing=0.2, handletextpad=0.2,
                   borderpad=0.1, frameon=False, loc='upper left',
                   alignment='center', ncols=2,
                   markerscale=1.8, columnspacing=1.)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.subplots_adjust(left=0.15, bottom=0.35)
    plt.ylim(-3., 4.)
    plt.savefig(figname, dpi=300)
    plt.close('all')

def compute_distances(df, class_column='class'):
    classes = df[class_column].unique()
    # initialize the distance matrix
    dmat = np.zeros((len(classes), len(classes)))
    print(classes)
    intra_dists = []
    # Compute In-Class Distances
    dist_metric = 'l2'
    for icls, cls in enumerate(classes):
        cls_data = df[df[class_column] == cls].drop(columns=[class_column]).values
        if len(cls_data) < 2:
            continue  # Skip classes with less than 2 samples
        distances = pairwise_distances(cls_data, metric=dist_metric)
        triu_indices = np.triu_indices_from(distances, k=1)
        dmat[icls,icls] = np.median(distances[triu_indices])
        intra_dists.extend(distances[triu_indices].tolist())
        print(np.median(distances))
    
    # Compute Inter-Class Distances
    inter_dists = []
    for i, cls1 in enumerate(classes):
        for j, cls2 in enumerate(classes[i+1:]):
            j2 = j + i + 1
            cls1_data = df[df[class_column] == cls1].drop(columns=[class_column]).values
            cls2_data = df[df[class_column] == cls2].drop(columns=[class_column]).values
            distances = pairwise_distances(cls1_data, cls2_data, metric=dist_metric)
            dij = np.median(distances)
            dmat[i,j2] = dij
            dmat[j2,i] = dij
            inter_dists.extend(distances.flatten().tolist())
    print(dmat)    

    return dmat, intra_dists, inter_dists


def comp_parc_and_full(parc_file, meta_file, local_file, me_file, target_region=''):
    np.random.seed(1024)
    random.seed(1024)

    parc = load_image(parc_file)
    if meta_file.endswith('xlsx'):
        meta = pd.read_excel(meta_file, index_col=0)
    else:
        meta = pd.read_csv(meta_file, index_col=0)
        # the file names in NeuroXiv is different from original names
        meta.index = [fname[9:-6] for fname in meta.index]

    dfl = pd.read_csv(local_file, index_col=0)
    dfme = pd.read_csv(me_file, index_col=0)
    # rename the me_feature names
    __ME_NAMES__ = [fn for fn in __FEAT_NAMES22__ if fn not in ('Nodes', 'SomaSurface', 'AverageDiameter', 'Surface')]
    dfme.drop(list(__ME_NAMES__), axis=1, inplace=True)
    mapper = {}
    for mf in __ME_NAMES__:
        mapper[f'{mf}_me'] = mf
    dfme.rename(columns=mapper, inplace=True)

    #feat_names = ['AverageContraction', 'AverageBifurcationAngleRemote',
    #              'HausdorffDimension', 'Bifurcations']

    # standardize
    standardize_features(dfl, __FEAT_NAMES22__)
    standardize_features(dfme, __ME_NAMES__)

    if target_region == 'CP':
        # This is for CP neurons
        # keep only the manual annotated CP neurons
        cp_neurons = meta[meta['Projection class'].isin(['CP_SNr', 'CP_GPe', 'CP_others'])]
        # get the parcellations
        coords = cp_neurons[['Soma_Z(CCFv3_1𝜇𝑚)', 'Soma_Y(CCFv3_1𝜇𝑚)', 'Soma_X(CCFv3_1𝜇𝑚)']] / 25.  # 25um
    else:
        # for hip neurons
        cp_neurons = meta[meta.region_name_ccf == target_region]
        cp_neurons = cp_neurons[cp_neurons.index.isin(dfl.index)]
        coords = cp_neurons[['z', 'y', 'x']] / 25. # to CCF-25 space

    # in parcellation
    zyx = np.floor(coords).astype(int).values
    # mirroring to left
    zdim = 456
    r_nz = np.nonzero(zyx[:,0] <= zdim/2)
    zyx[r_nz,0] = zdim - zyx[r_nz,0]
    # get the parcellations
    in_indices = np.nonzero(parc[zyx[:,0], zyx[:,1], zyx[:,2]] > 0)[0]
    print(in_indices.shape[0], coords.shape[0])
    # re-select the neurons
    in_zyx = zyx[in_indices]
    cp_neurons = cp_neurons.iloc[in_indices]
    # note, not tall the neurons are of full dendrites
    print(cp_neurons.shape)

    cp_parc = parc[in_zyx[:,0], in_zyx[:,1], in_zyx[:,2]] - 1 # start from 0

    if 1:   # pairwise similarity
        min_neurons = 15
        regs, rcnts = np.unique(cp_parc, return_counts=True)
        rmask = rcnts >= min_neurons
        regs_m = regs[rmask]

        parc_mask = np.isin(cp_parc, regs_m)
        cp_parc_m = cp_parc[parc_mask]
        neurons_m = cp_neurons[parc_mask].index

        # get the features
        dfl_m = dfl.loc[neurons_m][__FEAT_NAMES22__].copy()
        dfl_m['class'] = cp_parc_m
        # categorizing by classes
        dmat, intra_dists, inter_dists = compute_distances(dfl_m)
        # plot the distribution
        sns.histplot(intra_dists, color="blue", kde=False, label='Intra-dist', stat="density", bins=50, alpha=0.6)
        sns.histplot(inter_dists, color="red", kde=False, label='Inter-dist', stat="density", bins=50, alpha=0.6)
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Histogram of List1 and List2')
        plt.savefig(f'Intra_inter_distances_{target_region}.png', dpi=300)
        plt.close()


        sns.heatmap(dmat, cmap='seismic')
        plt.savefig(f'{target_region}.png', dpi=300); plt.close()
        print(dmat)


    # select neurons in target region
    r1_idx, r2_idx = 0, 12
    r7_mask = cp_parc == r1_idx
    r12_mask = cp_parc == r2_idx
    cp_parc_sub = cp_parc[r7_mask | r12_mask]
    cp_neurons_sub = cp_neurons[r7_mask | r12_mask]
    # 
    dfl_sub = dfl.loc[cp_neurons_sub.index]

    # get the in-region me files
    coords_me = dfme[['soma_z', 'soma_y', 'soma_x']] / 25.
    zyx_me = np.floor(coords_me).astype(int).values
    # mirroring to left
    zdim = 456
    r_nz = np.nonzero(zyx_me[:,0] <= zdim/2)
    zyx_me[r_nz,0] = zdim - zyx_me[r_nz,0]
    # get the parcellations
    parc_values_me = parc[zyx_me[:,0], zyx_me[:,1], zyx_me[:,2]]
    dfme_sub1 = dfme[parc_values_me == r1_idx+1]
    dfme_sub2 = dfme[parc_values_me == r2_idx+1]
    print(dfme_sub1.shape[0], dfme_sub2.shape[0])
    
    cp_parc_ints = cp_parc_sub.copy()
    cp_parc_ints[cp_parc_ints == r1_idx] = 0
    cp_parc_ints[cp_parc_ints == r2_idx] = 1

    compare_features(dfl_sub[cp_parc_sub == r1_idx], dfl_sub[cp_parc_sub == r2_idx], 
                     f'comp_features_full_dendrite_R{r1_idx+1}_R{r2_idx+1}.png')
    compare_features(dfme_sub1, dfme_sub2, 
                     f'comp_features_me_R{r1_idx+1}_R{r2_idx+1}.png')
    
    # overall
    emb_all, label_all, colors_all = clustering_on_umap(
                dfl.loc[neurons_m], feat_names=__FEAT_NAMES22__, nclusters=3, plot=True,
                figstr='tmp', precomputed_labels=cp_parc_m)

    emb_parc_l, labels_parc_l, colors_parc_l = clustering_on_umap(
                dfl_sub, feat_names=__FEAT_NAMES22__, nclusters=2, plot=True, 
                figstr='full_dendrite_by_parc', precomputed_labels=cp_parc_ints)
    emb_parc_m, labels_parc_m, colors_parc_m = clustering_on_umap(
                dfme_sub, feat_names=__ME_NAMES__, nclusters=2, plot=True, 
                figstr='me_by_parc', precomputed_labels=cp_parc_ints)

if __name__ == '__main__':
    if 1:
        # compare with existing neuron types
        rdict = {
            'CP': 672,
            'CA1': 382,
            'CA3': 463,
            'SUB': 502,
            'ProS': 484682470
        }

        target_region = 'CP'   # 'CA1', 'CA3', SUB, ProS
        target_regid = rdict[target_region]
        parc_file = f'../output_full_r671/parc_region{target_regid}.nrrd'
        if target_region == 'CP':
            meta_file = '../plotters/CP_single_morphologies/TableS6_Full_morphometry_1222.xlsx'
            local_file = '../plotters/CP_single_morphologies/cp_1876_dendrite_features.csv'
            me_file = '../data/mefeatures_100K_with_PCAfeatures3.csv'
        else:
            meta_file = '../plotters/whole-brain_projection/data/meta_hip.csv'
            local_file = '../plotters/hippocampus/ION_HIP/lm_features_d28_dendrites.csv'
            me_file = ''
        comp_parc_and_full(parc_file, meta_file, local_file, me_file, target_region=target_region)

