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
sys.path.append('../../../')
from config import __FEAT24D__

def get_highquality_subset(feature_file, flipLR=True, 
                names=['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']):
    df = pd.read_csv(feature_file, index_col=0)
    print(f'Initial number of recons: {df.shape[0]}')
    df = df[df.region_name.isin(names)]
    print(f'Number of filtered recons: {df.shape[0]}')
    
    df = df[df.isna().sum(axis=1) == 0]
    print(f'Number of non_na_recons: {df.shape[0]}')
    #assert df.isna().sum().sum() == 0
    if flipLR:
        # mirror right hemispheric points to left hemisphere
        zdim = 456 * 25
        zdim2 = zdim // 2
        nzi = np.nonzero(df['soma_z'] < zdim2)
        loci = df.index[nzi]
        df.loc[loci, 'soma_z'] = zdim - df.loc[loci, 'soma_z']

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
    def __init__(self, feature_file, topk=5, percentile=75):
        self.topk = topk
        self.df = get_highquality_subset(feature_file)
        self.radius = 166.34


    def calc_micro_env_features(self, mefeature_file):
        debug = False
        if debug: 
            self.df = self.df[:5000]
        
        df = self.df.copy()
        df_mef = df.copy()
        feat_names = __FEAT24D__
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


if __name__ == '__main__':
    if 1:
        feature_file = './lm_features_d28_dendrites.csv'
        mefile = f'./mefeatures_dendrites.csv'
        topk = 5
        
        mef = MEFeatures(feature_file, topk=topk)
        mef.calc_micro_env_features(mefile)
    
       

