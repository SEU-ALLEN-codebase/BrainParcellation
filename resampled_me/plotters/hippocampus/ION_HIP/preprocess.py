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
from swc_handler import get_soma_from_swc
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree



def aggregate_information(swc_dir, gf_file, reg_file, outfile):
    DEBUG = False

    if not os.path.exists(reg_file):
        atlas = load_image(MASK_CCF25_FILE)
        dim = np.array(atlas.shape)
        ana_tree = parse_ana_tree()
        # get the soma position and brain region
        rinfos = []
        for swc_file in glob.glob(os.path.join(swc_dir, '*.swc')):
            name = os.path.split(swc_file)[-1][:-4]
            spos = np.array(list(map(float, get_soma_from_swc(swc_file)[2:5])))
            spos_ccf = np.floor(spos / 25.).astype(int)[::-1]   # to zyx
            # get the region based on its coordinate
            if (spos_ccf > dim).sum() > 0:
                region_id = 0
                region_name = ''
            else:
                region_id = atlas[spos_ccf[0], spos_ccf[1], spos_ccf[2]]
                region_name = ana_tree[region_id]['acronym']
            rinfo = [name, *spos, region_id, region_name]
            rinfos.append(rinfo)
            
            #if len(rinfos) > 10:
            #    break

        df_soma = pd.DataFrame(rinfos, columns=['Name', 'soma_x', 'soma_y', 'soma_z', 'region_id', 'region_name'])
        df_soma.set_index('Name', inplace=True)
        df_soma.to_csv(reg_file)
    else:
        df_soma = pd.read_csv(reg_file, index=0)
    print('Soma information extraction done!')
            

    df_feat = pd.read_csv(gf_file, index_col=0) # d22

    # ,soma_x,soma_y,soma_z,dataset_name,brain_id,pc11,pc12,pc13,pca_vr1,pca_vr2,pca_vr3,region_id_r671,region_name_r671,region_id_r316,region_name_r316,brain_structure
    df = df_feat.merge(df_soma, left_index=True, right_index=True)

    # get additional meta information and features
    # load the 
    new_cols = []
    t0 = time.time()
    for irow, row in df.iterrows():
        # estimate the isotropy
        swcfile = os.path.join(swc_dir, f'{irow}.swc')
        coords = np.genfromtxt(swcfile, usecols=(2,3,4))
        pca = decomposition.PCA()
        pca.fit(coords)
        new_cols.append((*pca.components_[0], *pca.explained_variance_ratio_))

        if len(new_cols) % 100 == 0:
            print(f'[{len(new_cols)}]: {time.time() - t0:.2f} seconds')
            if DEBUG: break

    tmp_index = df.index[:len(new_cols)] if DEBUG else df.index
    new_cols = pd.DataFrame(new_cols, 
        columns=['pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3'],
        index=tmp_index
    ).astype({'pc11': np.float32,
            'pc12': np.float32, 'pc13': np.float32, 'pca_vr1': np.float32, 'pca_vr2': np.float32, 
            'pca_vr3': np.float32})

    tmp_df = df[:len(new_cols)] if DEBUG else df
    df = df.merge(new_cols, left_index=True, right_index=True)
    
    df.to_csv(outfile, float_format='%g')
    
    
    


if __name__ == '__main__':
    swc_dir = '/PBshare/SEU-ALLEN/Users/Sujun/ION_Hip_CCFv3_crop'
    gf_file = 'gf_ion_hip.csv'
    reg_file = 'soma_region.csv'
    outfile = 'lm_features_d28.csv'

    aggregate_information(swc_dir, gf_file, reg_file, outfile)

    
