#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : micro_env_features.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-18
#   Description  : 
#
#================================================================
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree

from swc_handler import get_soma_from_swc
from math_utils import min_distances_between_two_sets

import sys
sys.path.append('../common_lib')
from configs import __FEAT_NAMES__, __FEAT_ALL__

def get_highquality_subset(feature_file, filter_file):
    df = pd.read_csv(feature_file, index_col=0)
    print(f'Initial number of recons: {df.shape[0]}')
    fl = pd.read_csv(filter_file, names=['Name'])
    names = [n[:-9] for n in fl.Name]
    df = df[df.index.isin(names)]
    print(f'Number of filtered recons: {df.shape[0]}')

    assert df.isna().sum().sum() == 0
    return df

def estimate_radius(lmf, topk=5, percentile=50):
    spos = lmf[['soma_x', 'soma_y', 'soma_z']]
    topk_d = min_distances_between_two_sets(spos, spos, topk=topk+1, reciprocal=False)
    topk_d = topk_d[:,-1]
    pp = [0, 25, 50, 75, 100]
    pcts = np.percentile(topk_d, pp)
    print(f'top{topk} threshold percentile: {pcts}')    #[ 28.34184539  96.11359243 124.45518049 166.36041146 773.71830979]
    pct = np.percentile(topk_d, percentile)
    print(f'Selected threshold by percentile[{percentile}] = {pct:.2f} um')
    
    return pct
    
    

class MEFeatures:
    def __init__(self, feature_file, filter_file, topk=5, percentile=75):
        self.topk = topk
        self.df = get_highquality_subset(feature_file, filter_file)
        self.radius = estimate_radius(self.df, topk=topk, percentile=percentile)


    def calc_micro_env_features(self, mefeature_file):
        debug = False
        if debug: 
            self.df = self.df[:5000]
        
        df = self.df.copy()
        df_mef = df.copy()
        feat_names = __FEAT_NAMES__ + ['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3']
        mefeat_names = [f'{fn}_me' for fn in feat_names]

        df_mef[mefeat_names] = 0.
    
        # we should pre-normalize each feature for topk extraction
        feats = df.loc[:, feat_names]
        feats = (feats - feats.mean()) / (feats.std() + 1e-10)
        df[feat_names] = feats

        spos = df[['soma_x', 'soma_y', 'soma_z']]
        print(f'--> Extracting the neurons within radius for each neuron')
        # using kdtree to find out neurons within radius
        spos_kdt = KDTree(spos, leaf_size=2)
        in_radius_neurons = spos_kdt.query_radius(spos, self.radius, return_distance=True)

        # iterate over all samples
        t0 = time.time()
        for i, indices, dists in zip(range(spos.shape[0]), *in_radius_neurons):
            f0 = feats.iloc[i]  # feature for current neuron
            fir = feats.iloc[indices]   # features for in-range neurons
            fdists = np.linalg.norm(f0 - fir, axis=1)
            # select the topK most similar neurons for feature aggregation
            k = min(self.topk, fir.shape[0]-1)
            idx_topk = np.argpartition(fdists, k)[:k+1]
            # map to the original index space
            topk_indices = indices[idx_topk]
            topk_dists = dists[idx_topk]

            # get the average features
            swc = df_mef.index[i]
            # spatial-tuned features
            dweights = np.exp(-topk_dists/self.radius)
            dweights /= dweights.sum()
            values = self.df.iloc[topk_indices][feat_names] * dweights.reshape(-1,1)

            if len(topk_indices) == 1:
                df_mef.loc[swc, mefeat_names] = values.to_numpy()[0]
            else:
                df_mef.loc[swc, mefeat_names] = values.sum().to_numpy()

            if i % 1000 == 0:
                print(f'[{i}]: time used: {time.time()-t0:.2f} seconds')
            
        df_mef.to_csv(mefeature_file, float_format='%g')


def calc_regional_mefeatures(mefeature_file, rmefeature_file, region_num=316):
    mef = pd.read_csv(mefeature_file, index_col=0)
    print(f'Feature shape: {mef.shape}')
    # calculate the mean and average features
    rkey = f'region_id_r{region_num}'
    rnkey = f'region_name_r{region_num}'
    regions = np.unique(mef[rkey])
    output = []
    index = []
    for region in regions:
        region_index = mef.index[mef[rkey] == region]
        feat = mef.loc[region_index, __FEAT_ALL__]
        rid = mef.loc[region_index[0], rkey]
        rname = mef.loc[region_index[0], rnkey]
        struct = mef.loc[region_index[0], 'brain_structure']
        fmean = feat.mean().to_numpy().tolist()
        fstd = feat.std().to_numpy().tolist()
        index.append(rid)
        output.append([rname, struct, len(region_index), *fmean, *fstd])

    columns = [rnkey, 'brain_structure', 'NumRecons']
    columns.extend([f'{fn}_mean' for fn in __FEAT_ALL__])
    columns.extend([f'{fn}_std' for fn in __FEAT_ALL__])
    rmef = pd.DataFrame(output, index=index, columns=columns)
    rmef.to_csv(rmefeature_file, float_format='%g')

       

if __name__ == '__main__':
    if 1:
        feature_file = './data/lm_features_d28.csv'
        filter_file = '../evaluation/data/final_filtered_swc.txt'
        mefile = f'./data/mefeatures_100K.csv'
        topk = 5
        
        mef = MEFeatures(feature_file, filter_file=filter_file, topk=topk)
        mef.calc_micro_env_features(mefile)
    
       
    if 0:
        nodes_range = (300, 1500)
        mefeature_file = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_withoutNorm_statis.csv'
        rmefeature_file = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_regional.csv'
        topk = 5
        
        calc_regional_mefeatures(mefeature_file, rmefeature_file)

