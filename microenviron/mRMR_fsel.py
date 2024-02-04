#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : mRMR_fsel.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-11
#   Description  : 
#
#================================================================

import numpy as np
import pandas as pd
import pymrmr

def load_features(ffile):
    __FEATS__ = 'Stems_me,Bifurcations_me,Branches_me,Tips_me,OverallWidth_me,OverallHeight_me,OverallDepth_me,Length_me,Volume_me,MaxEuclideanDistance_me,MaxPathDistance_me,MaxBranchOrder_me,AverageContraction_me,AverageFragmentation_me,AverageParent-daughterRatio_me,AverageBifurcationAngleLocal_me,AverageBifurcationAngleRemote_me,HausdorffDimension_me,pca_vr1_me,pca_vr2_me,pca_vr3_me'.split(',')

    df = pd.read_csv(ffile, index_col=0)
    df = df[['region_id_r316', *__FEATS__]]
    #df['pca_vr_diff_me'] = df['pca_vr1_me'].to_numpy() - df['pca_vr3_me'].to_numpy()
    #feat_names = __FEATS__ + ['pca_vr_diff_me']
    feat_names = __FEATS__

    tmp = df[feat_names]
    df.loc[:, feat_names] = (df.loc[:, feat_names] - tmp.mean()) / (tmp.std() + 1e-10)
    regions = df['region_id_r316']
    uregions = np.unique(regions)
    print(f'#classes: {len(uregions)}')

    rdict = dict(zip(uregions, range(len(uregions))))
    rindices = [rdict[rname] for rname in regions]

    df.loc[:, 'region_id_r316'] = rindices
    feats = pymrmr.mRMR(df, 'MIQ', 5)

    print(feats)

if __name__ == '__main__':
    ffile = './data/mefeatures_100K.csv'
    load_features(ffile)

