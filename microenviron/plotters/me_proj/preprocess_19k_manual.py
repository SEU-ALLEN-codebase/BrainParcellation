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
from swc_handler import get_soma_from_swc



def aggregate_information(swc_dir, gf_file, outfile):
    DEBUG = False
    datasets = {
        'hip': 'ion_hip_2um',
        'pfc': 'ion_pfc_2um',
        'mouselight': 'mouselight_2um',
        'seuallen': 'seu-allen1876_2um',
    }

    df_feat = pd.read_csv(gf_file, index_col=0) # d22

    # ,soma_x,soma_y,soma_z,dataset_name,brain_id,pc11,pc12,pc13,pca_vr1,pca_vr2,pca_vr3,region_id_r671,region_name_r671,region_id_r316,region_name_r316,brain_structure

    # get additional meta information and features
    # load the 
    ana_tree = parse_ana_tree()
    ccf_atlas = load_image(MASK_CCF25_FILE)

    new_cols = []
    t0 = time.time()
    for irow, row in df_feat.iterrows():
        # get the SWC file
        if irow.startswith('Mouse'):
            data_name = 'mouselight'
            swcfile = os.path.join(swc_dir, datasets[data_name], f'{irow}.swc')
        elif irow.startswith('ION'):
            data_name = 'hip'
            swcfile = os.path.join(swc_dir, datasets[data_name], f'{irow}.swc')
            if not os.path.exists(swcfile):
                data_name = 'pfc'
                swcfile = os.path.join(swc_dir, datasets[data_name], f'{irow}.swc')
        else:
            data_name = 'seuallen'
            swcfile = os.path.join(swc_dir, datasets[data_name], f'{irow}.swc')
        # get the soma location
        sloc = np.array(list(map(float, get_soma_from_swc(swcfile)[2:5])))
        sx, sy, sz = sloc
        sxi, syi, szi = np.floor(sloc / 25.).astype(int)    # voxel coordinate in CCF
        # load the CCF region
        try:
            rid = ccf_atlas[szi, syi, sxi]
            rname = ana_tree[rid]['acronym']
        except:
            rid = 0
            rname = ''

        # estimate the isotropy
        coords = np.genfromtxt(swcfile, usecols=(2,3,4))
        pca = decomposition.PCA()
        pca.fit(coords)
        new_cols.append((data_name, rname, sx, sy, sz, rid, *pca.components_[0], *pca.explained_variance_ratio_))

        if len(new_cols) % 100 == 0:
            print(f'[{len(new_cols)}]: {time.time() - t0:.2f} seconds')
            if DEBUG: break

    tmp_index = df_feat.index[:len(new_cols)] if DEBUG else df_feat.index
    new_cols = pd.DataFrame(new_cols, 
        columns=['dataset', 'region_name_r316', 'soma_x', 'soma_y', 'soma_z', 
                 'region_id_r316', 'pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3'],
        index=tmp_index
    ).astype({'dataset': str, 'region_name_r316':str, 'soma_x': float, 'soma_y': float, 'soma_z': float, 
            'region_id_r316': int, 'pc11': np.float32,
            'pc12': np.float32, 'pc13': np.float32, 'pca_vr1': np.float32, 'pca_vr2': np.float32, 
            'pca_vr3': np.float32})

    tmp_df = df_feat[:len(new_cols)] if DEBUG else df_feat
    df = df_feat.merge(new_cols, left_index=True, right_index=True)
    
    df.to_csv(outfile, float_format='%g')
    
    
    


if __name__ == '__main__':
    dtype = 'dendrites'
    swc_dir = f'/data/lyf/data/fullNeurons/all_neurons_{dtype}'
    gf_file = f'data/gf_all_2w_2um_{dtype}.csv'
    outfile = f'data/lm_features_{dtype}.csv'
    aggregate_information(swc_dir, gf_file, outfile)

    
