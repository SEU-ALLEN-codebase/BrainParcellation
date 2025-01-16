##########################################################
#Author:          Yufeng Liu
#Create time:     2025-01-11
#Description:     utilities for generating reponse materials at the round#1 phase
##########################################################
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from config import load_features

def pairwise_distance_correlation(feats1, feats2, figname, xlabel=None, ylabel=None, rpoint=1):
    # estimate the pairwise distances
    dists1 = pdist(feats1)
    dists2 = pdist(feats2)
    # select a subset for verification
    random.seed(1024)   # ensure duplicate
    nsel = 5000
    selids = random.sample(range(len(dists1)), nsel)
    sub_dists1 = dists1[selids]
    sub_dists2 = dists2[selids]
    
    # fitting the data
    if xlabel is None:
        xlabel = f'Full set (#={feats1.shape[1]})'
    if ylabel is None:
        ylabel = f'Subset (#={feats2.shape[1]})'
    df_dists = pd.DataFrame(np.stack([sub_dists1, sub_dists2]).transpose(), columns=(xlabel, ylabel))
    sns.set_theme(style='ticks', font_scale=1.6)
    plt.figure(figsize=(6,6))
    ax = sns.regplot(df_dists, x=xlabel, y=ylabel, 
                scatter_kws={'s':rpoint, 'color':'black'},
                line_kws={'color':'magenta', 'alpha':0.3})
    # estimate the Pearson correlation between them
    presult = pearsonr(sub_dists1, sub_dists2)
    ax.text(.1, .75, f'Correlation={presult.statistic:.2f}\np-value={presult.pvalue:.1f}',
                fontsize=20, transform=ax.transAxes, color='black')

    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.xticks([2,4,6,8,10,12,14])
    plt.yticks([2,4,6,8,10,12,14])
    #plt.axis('equal')
    plt.savefig(figname, dpi=300)
    plt.close()


def estimate_feature_redundancy(mefile, scale=25.):
    df, fnames = load_features(mefile, scale=scale, feat_type='full')
    # use the hippocampus neurons for illustration
    hip_names = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
    hip_neurons = df[df.region_name_r671.isin(hip_names)]

    hip_feats1 = hip_neurons[fnames]
    # remove the dimension-related features
    #rm_fnames = ['OverallWidth_me', 'OverallHeight_me', 'OverallDepth_me', 'MaxEuclideanDistance_me']
    rm_fnames = ['OverallWidth_me', 'OverallHeight_me', 'OverallDepth_me', 'MaxEuclideanDistance_me', 'MaxPathDistance_me']
    fnames2 = [fname for fname in fnames if fname not in rm_fnames]
    hip_feats2 = hip_neurons[fnames2]

    figname = 'correlation_of_different_feature_sets.png'
    pairwise_distance_correlation(hip_feats1, hip_feats2, figname)


def estimate_me_parameters(mefile1, mefile2, figname, xlabel, ylabel, scale=25., rpoint=1):
    df1, fnames1 = load_features(mefile1, scale=scale, feat_type='full')
    df2, fnames2 = load_features(mefile2, scale=scale, feat_type='full')
    
    # use hippocampus for illustration
    hip_names = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
    hip_mask = df1.region_name_r671.isin(hip_names)
    hip_feats1 = df1[hip_mask][fnames1]
    hip_feats2 = df2[hip_mask][fnames2]

    pairwise_distance_correlation(hip_feats1, hip_feats2, figname, xlabel, ylabel, rpoint=rpoint)


if __name__ == '__main__':
    mefile = '../data/mefeatures_100K_with_PCAfeatures3.csv'

    if 1:
        #estimate_feature_redundancy(mefile, scale=25.)
        mefile1 = '../data/mefeatures_100K_radius137.28.csv'
        figname = 'me_radius137.28_vs_radius166.36.png'
        xlabel = 'radius=166.36 μm'
        ylabel = 'radius=137.28 μm'
        estimate_me_parameters(mefile, mefile1, figname, xlabel, ylabel, rpoint=5)

