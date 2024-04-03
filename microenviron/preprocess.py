#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : preprocess.py
#   Author       : Yufeng Liu
#   Date         : 2023-01-17
#   Description  : Aggregate the informations
#
#================================================================

import os
import glob
import time
import re
import numpy as np
import pandas as pd
from sklearn import decomposition

from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree



def aggregate_information(swc_dir, gf_file, reg_file, soma_file, outfile):
    DEBUG = False

    df_feat = pd.read_csv(gf_file, index_col=0) # d22
    df_reg = pd.read_csv(reg_file, usecols=(1,2,3), index_col=1, skiprows=1, names=['brain_id', 'Name', 'region_name_r316'])
    df_soma = pd.read_csv(soma_file, index_col=0, sep=' ', names=['Name', 'soma_x', 'soma_y', 'soma_z'])

    # ,soma_x,soma_y,soma_z,dataset_name,brain_id,pc11,pc12,pc13,pca_vr1,pca_vr2,pca_vr3,region_id_r671,region_name_r671,region_id_r316,region_name_r316,brain_structure
    # processing df_reg
    df_reg_new_indices = [iname[:-9] for iname in df_reg.index]
    df_reg.set_index(pd.Index(df_reg_new_indices), inplace=True)
    # merge the brain and region information
    df = df_feat.merge(df_reg, left_index=True, right_index=True)
    # processing df_soma
    df_soma_new_indices = [iname[:-5] for iname in df_soma.index]
    df_soma.set_index(pd.Index(df_soma_new_indices), inplace=True)
    # merge
    df = df.merge(df_soma, left_index=True, right_index=True)

    # get additional meta information and features
    # load the 
    ana_tree = parse_ana_tree(keyname='name')
    new_cols = []
    t0 = time.time()
    for irow, row in df.iterrows():
        rname = row.region_name_r316
        try:
            rid = ana_tree[rname]['id']
        except KeyError:
            #print(rname)
            rid = 0
        # estimate the isotropy
        swcfile = os.path.join(swc_dir, str(row.brain_id), f'{irow}_stps.swc')
        coords = np.genfromtxt(swcfile, usecols=(2,3,4))
        pca = decomposition.PCA()
        pca.fit(coords)
        new_cols.append((rid, *pca.components_[0], *pca.explained_variance_ratio_))

        if len(new_cols) % 100 == 0:
            print(f'[{len(new_cols)}]: {time.time() - t0:.2f} seconds')
            if DEBUG: break

    tmp_index = df.index[:len(new_cols)] if DEBUG else df.index
    new_cols = pd.DataFrame(new_cols, 
        columns=['region_id_r316', 'pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3'],
        index=tmp_index
    ).astype({'region_id_r316': int, 'pc11': np.float32,
            'pc12': np.float32, 'pc13': np.float32, 'pca_vr1': np.float32, 'pca_vr2': np.float32, 
            'pca_vr3': np.float32})

    tmp_df = df[:len(new_cols)] if DEBUG else df
    df = df.merge(new_cols, left_index=True, right_index=True)
    
    df.to_csv(outfile, float_format='%g')
    
    
    


if __name__ == '__main__':
    swc_dir = '/PBshare/SEU-ALLEN/Users/Sujun/230k_organized_folder/cropped_100um/'
    gf_file = '/PBshare/SEU-ALLEN/Users/Sujun/230k_organized_folder/analysis/gf_179k_crop.csv'
    reg_file = '../evaluation/data/179k_soma_region.csv'
    soma_file = '../evaluation/data/179k_somalist.txt'
    outfile = 'data/lm_features_d28.csv'
    aggregate_information(swc_dir, gf_file, reg_file, soma_file, outfile)

    
