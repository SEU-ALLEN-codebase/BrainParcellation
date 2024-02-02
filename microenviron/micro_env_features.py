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

from swc_handler import get_soma_from_swc
from math_utils import min_distances_between_two_sets

import sys
sys.path.append('../common_lib')
from configs import __FEAT_NAMES__, __FEAT_ALL__

def get_highquality_subset(feature_file, nodes_range=(500,1500), min_num_recons=0, remove_nan=True, region_num=316):
    df = pd.read_csv(feature_file, index_col=0)
    print(f'Initial number of recons: {df.shape[0]}')
    if remove_nan:
        err_f = ~(df.isna().sum(axis=1).astype(bool))
        df = df[err_f]
    print(f'Number after remove_nan: {df.shape[0]}')

    # keep regions with at leat `min_num_recons` neurons
    rkey = f'region_id_r{region_num}'
    regions, counts = np.unique(df[rkey], return_counts=True)
    abd_regions = regions[counts >= min_num_recons]
    df = df[df[rkey].isin(abd_regions)]
    print(f'Number after removing regions less thant {min_num_recons}: {df.shape[0]}')

    # filter out the regions with number of nodes out of range `nodes_range`
    nmin, nmax = nodes_range
    df = df[(df['Nodes'] >= nmin) & (df['Nodes'] <= nmax)]
    print(f'Number of samples after nodes pruning: {df.shape[0]}')
    return df

def estimate_radius(lmf, topk=5, percentile=50):
    spos = lmf[['soma_x', 'soma_y', 'soma_z']]
    topk_d = min_distances_between_two_sets(spos, spos, topk=topk+1, reciprocal=False)
    topk_d = topk_d[:,-1]
    pp = [0, 25, 50, 75, 100]
    pcts = np.percentile(topk_d, pp)
    print(f'top{topk} threshold percentile: {pcts}') # [3.16, 8.32, 9.95, 12.09, 75.40]
    pct = np.percentile(topk_d, percentile)
    print(f'Selected threshold by percentile[{percentile}] = {pct}')
    
    return pct
    
    

class MEFeatures:
    def __init__(self, feature_file, region_num='316', nodes_range=(500,1500), min_num_recons=0, topk=5):
        self.region_num = region_num    # not the actual number of regions
        self.topk = topk
        self.df = get_highquality_subset(feature_file, nodes_range, min_num_recons, True, region_num)
        self.radius = estimate_radius(self.df, topk=topk)


    def calc_micro_env_features(self, mefeature_file):
        debug = False
        if debug: 
            self.df = self.df[:5000]
        
        df = self.df.copy()
        df_mef = df.copy()
        feat_names = __FEAT_NAMES__ + ['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3']
        mefeat_names = [f'{fn}_me' for fn in feat_names]

        df_mef[mefeat_names] = 0
    
        # we should pre-normalize each feature for topk extraction
        feats = df.loc[:, feat_names]
        feats = (feats - feats.mean()) / (feats.std() + 1e-10)
        df[feat_names] = feats

        spos = df[['soma_x', 'soma_y', 'soma_z']]
        print(f'--> Estimating the pairwise distance of soma')
        pdists = distance_matrix(spos, spos)
        print(f'--> Estimating the pairwise distance of features')
        fdists = distance_matrix(feats, feats)
        print('--> Geting mask')
        pos_mask = pdists < self.radius
        # iterate over all samples
        for i in range(pos_mask.shape[0]):
            mi = pos_mask[i]
            fi = fdists[i]
            di = pdists[i]
            # in-radius neurons
            mi_pidx = mi.nonzero()[0]
            fi_p = fi[mi_pidx]  # in-radius features
            # select the top-ranking neurons
            k = min(self.topk, len(mi_pidx)-1)
            #print(k, len(mi_pidx))
            idx_topk = np.argpartition(fi_p, k)[:k+1]
            # convert to the original index space
            orig_idx = mi_pidx[idx_topk]
            # get the average features
            swc = df_mef.index[i]
            # spatial-tuned features
            
            dweights = np.exp(-di[orig_idx]/self.radius)
            dweights /= dweights.sum()
            values = self.df.iloc[orig_idx][feat_names] * dweights.reshape(-1,1)

            if len(orig_idx) == 1:
                df_mef.loc[swc, mefeat_names] = values.to_numpy()[0]
            else:
                df_mef.loc[swc, mefeat_names] = values.sum().to_numpy()

            if i % 100 == 0:
                print(i)
            
        df_mef.to_csv(mefeature_file, float_format='%g')

    def calc_micro_env_features_with_statis(self, mefeature_file, min_neighbors=3):
        debug = False
        if debug: 
            self.df = self.df[:5000]
        
        df = self.df.copy()
        df_mef = self.df.copy()
        df_mef[__FEAT_ALL__] = 0
        feat_names = __FEAT_NAMES__ 

        swcs = []
    
        # we should pre-normalize each feature for topk extraction
        feats = df.loc[:, feat_names]
        feats = (feats - feats.mean()) / (feats.std() + 1e-10)
        df[feat_names] = feats

        spos = df[['soma_x', 'soma_y', 'soma_z']]
        print(f'--> Estimating the pairwise distance of soma')
        pdists = distance_matrix(spos, spos)
        print(f'--> Estimating the pairwise distance of features')
        fdists = distance_matrix(feats, feats)
        print('--> Geting mask')
        pos_mask = pdists < self.radius
        # iterate over all samples
        for i in range(pos_mask.shape[0]):
            mi = pos_mask[i]
            fi = fdists[i]
            di = pdists[i]
            # in-radius neurons
            mi_pidx = mi.nonzero()[0]
            if len(mi_pidx) < min_neighbors:
                print('Not enough neurons')
                continue
            fi_p = fi[mi_pidx]  # in-radius features
            # select the top-ranking neurons
            k = min(self.topk, len(mi_pidx)-1)
            #print(k, len(mi_pidx))
            idx_topk = np.argpartition(fi_p, k)[:k+1]
            # convert to the original index space
            orig_idx = mi_pidx[idx_topk]
            # get the average features
            swc = df_mef.index[i]
            # spatial-tuned features
            dweights = np.exp(-di[orig_idx]/self.radius)
            dweights /= dweights.sum()
            values = self.df.iloc[orig_idx][feat_names] * dweights.reshape(-1,1)

            vmean = values.mean().to_numpy()
            vmedian = values.median().to_numpy()
            vstd = values.std().to_numpy()
            vall = np.hstack((vmean, vmedian, vstd))
            
            swcs.append(swc)
            df_mef.loc[swc, __FEAT_ALL__] = vall

            if i % 100 == 0:
                print(i)

        df_mef = df_mef[df_mef.index.isin(swcs)]
        # normalize
        tmp = df_mef.loc[:, __FEAT_ALL__]
        tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
        df_mef.loc[:, __FEAT_ALL__] = tmp

            
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
        nodes_range = (300, 1500)
        feature_file = './data/lm_features_d22_all.csv'
        mefile = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_withoutNorm.csv'
        topk = 5
        
        mef = MEFeatures(feature_file, nodes_range=nodes_range, topk=topk)
        mef.calc_micro_env_features(mefile)
    
    if 0:   # with statis micro-environ features
        nodes_range = (300, 1500)
        feature_file = './data/lm_features_d22_all.csv'
        mefile = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_withoutNorm_statis.csv'
        topk = 5
        min_neighbors = 3
        
        mef = MEFeatures(feature_file, nodes_range=nodes_range, topk=topk)
        mef.calc_micro_env_features_with_statis(mefile, min_neighbors)
        
    if 0:
        nodes_range = (300, 1500)
        mefeature_file = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_withoutNorm_statis.csv'
        rmefeature_file = f'./data/micro_env_features_nodes{nodes_range[0]}-{nodes_range[1]}_regional.csv'
        topk = 5
        
        calc_regional_mefeatures(mefeature_file, rmefeature_file)

