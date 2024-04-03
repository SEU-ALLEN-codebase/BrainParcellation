#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : feat_sel.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-11
#   Description  : 
#
#================================================================
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pymrmr
import pickle

from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE, ANATOMY_TREE_FILE
from anatomy.anatomy_core import parse_ana_tree

def load_features(ffile, only_features=False):
    __FEATS__ = 'Stems_me,Bifurcations_me,Branches_me,Tips_me,OverallWidth_me,OverallHeight_me,OverallDepth_me,Length_me,Volume_me,MaxEuclideanDistance_me,MaxPathDistance_me,MaxBranchOrder_me,AverageContraction_me,AverageFragmentation_me,AverageParent-daughterRatio_me,AverageBifurcationAngleLocal_me,AverageBifurcationAngleRemote_me,HausdorffDimension_me,pca_vr1_me,pca_vr2_me,pca_vr3_me'.split(',')

    df = pd.read_csv(ffile, index_col=0)
    if only_features:
        df = df[[*__FEATS__]]
    else:
        df = df[['region_id_r316', *__FEATS__]]
    #df['pca_vr_diff_me'] = df['pca_vr1_me'].to_numpy() - df['pca_vr3_me'].to_numpy()
    #feat_names = __FEATS__ + ['pca_vr_diff_me']
    feat_names = __FEATS__

    tmp = df[feat_names]
    df.loc[:, feat_names] = (df.loc[:, feat_names] - tmp.mean()) / (tmp.std() + 1e-10)

    if not only_features:
        regions = df['region_id_r316']
        uregions = np.unique(regions)
        print(f'#classes: {len(uregions)}')

        rdict = dict(zip(uregions, range(len(uregions))))
        rindices = [rdict[rname] for rname in regions]

        df.loc[:, 'region_id_r316'] = rindices
    return df
    

def exec_mrmr(ffile):
    df = load_features(ffile, only_features=False)
    method='MIQ'
    topk=5
    feats = pymrmr.mRMR(df, method, topk)
    print(feats)
    return feats

def exec_pca(ffile, fpca_file):
    df = load_features(ffile, only_features=True)
    pca = PCA(n_components=3, whiten=False)
    pca_feat3 = pca.fit_transform(df)
    vr1, vr2, vr3 = pca.explained_variance_ratio_
    print(f'Variance ratios: {vr1:.4f}, {vr2:.4f}, {vr3:.4f}')

    if os.path.exists(fpca_file):
        os.system(f'rm {fpca_file}')
    
    dfo = pd.read_csv(ffile, index_col=0)
    # append the features to the original feat_file
    dfo['pca_feat1'] = pca_feat3[:,0]
    dfo['pca_feat2'] = pca_feat3[:,1]
    dfo['pca_feat3'] = pca_feat3[:,2]
    # we should also annotate the r671 regions
    mask = load_image(MASK_CCF25_FILE)
    ana_tree = parse_ana_tree()
    # check the existing region indices are correct
    coords = dfo[['soma_z', 'soma_y', 'soma_x']] / 25.
    coords_floor = np.floor(coords).astype(int)
    r671_ids = mask[coords_floor['soma_z'], coords_floor['soma_y'], coords_floor['soma_x']]
    r671_names = []
    for idx in r671_ids:
        if idx == 0:
            r671_names.append('')
        else:
            r671_names.append(ana_tree[idx]['acronym'])
    dfo['region_name_r671'] = r671_names
    dfo['region_id_r671'] = r671_ids

    # I would like to also correct the R314 region
    r314_mask = load_image(MASK_CCF25_R314_FILE)
    

    with open(fpca_file, 'a') as fp:
        fp.write(f'# explained_variance_ratios: {vr1:.4f}, {vr2:.4f}, {vr3:.4f}\n')
        dfo.to_csv(fp)

if __name__ == '__main__':
    ffile = './data/mefeatures_100K.csv'
    #exec_mrmr(ffile)
    exec_pca(ffile, f'{os.path.splitext(ffile)[0]}_with_PCAfeatures3.csv')

