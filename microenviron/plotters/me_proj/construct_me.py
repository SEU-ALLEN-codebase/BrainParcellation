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

from math_utils import min_distances_between_two_sets

import sys
sys.path.append('../../../common_lib')
from configs import __FEAT_NAMES__, __FEAT_ALL__

def get_highquality_subset(feature_file, filter_file):
    df = pd.read_csv(feature_file, index_col=0)
    print(f'Initial number of recons: {df.shape[0]}')
    fl = pd.read_csv(filter_file, names=['Name'])
    names = fl.Name #[n[:-9] for n in fl.Name]
    df = df[df.index.isin(names)]
    print(f'Number of filtered recons: {df.shape[0]}')

    df = df[df.isna().sum(axis=1) == 0]
    print(f'Number of non_na_recons: {df.shape[0]}')
    #assert df.isna().sum().sum() == 0
    return df

   

class MEFeatures:
    def __init__(self, feature_file, topk=5, radius=166.36):
        self.topk = topk
        self.df = pd.read_csv(feature_file, index_col=0)
        self.radius = radius


    def calc_micro_env_features(self, mefeature_file, min_k=2):
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
        keep_indices = []
        for i, indices, dists in zip(range(spos.shape[0]), *in_radius_neurons):
            if indices.shape[0] <= min_k:
                continue
            else:
                keep_indices.append(i)

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

            df_mef.loc[swc, mefeat_names] = values.sum().to_numpy()

            if i % 1000 == 0:
                print(f'[{i}]: time used: {time.time()-t0:.2f} seconds')
            
        # keep only ME with sufficient neurons nearby
        df_mef = df_mef.iloc[keep_indices]
        df_mef.to_csv(mefeature_file, float_format='%g')

      

if __name__ == '__main__':
    if 1:
        dtype = 'dendrites100'
        feature_file = f'./data/lm_features_{dtype}.csv'
        mefile = f'./data/mefeatures_{dtype}.csv'
        topk = 5
        
        mef = MEFeatures(feature_file, topk=topk)
        mef.calc_micro_env_features(mefile)
       

