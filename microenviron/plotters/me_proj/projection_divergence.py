##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-13
#Description:               
##########################################################
import os
import sys
import glob
import pickle
import time
import numpy as np
import pandas as pd

from anatomy.anatomy_config import SALIENT_REGIONS, MASK_CCF25_FILE
from file_io import load_image

sys.path.append('../..')
from config import get_me_ccf_mapper

def spatial_variance(points, order=2):
    centroid = np.mean(points, axis=0)
    distances = np.sum((points - centroid) ** 2, axis=1)
    variance = np.mean(distances)
    if order == 1:
        return np.sqrt(variance)
    
    return variance


def update_spatial_variance(centroid, variance, count, new_points):
    # update the number of points, and the centroid
    new_count = count + len(new_points)
    new_centroid = (centroid * count + np.sum(new_points, axis=0)) / new_count
    # calculate the variance
    dcentroid = centroid - new_centroid
    new_distances = np.sum((new_points - new_centroid) ** 2, axis=1)
    new_variance = (variance*count + np.sum(dcentroid**2 * count) + np.sum(new_distances)) / new_count

    return new_centroid, new_variance, new_count


def projection_divergence(swc_dir, meta_file, out_csv, me_atlas_file, me2ccf_file, only_tip=False):
    # Load the meta information
    meta = pd.read_csv(meta_file, index_col=0)
    n_ccf_cnt = 20
    n_me_cnt = 10
    #regions = ['CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 
    #           'PAR', 'POST', 'PRE', 'ProS', 'SUB']
    regions, rcnts = np.unique(meta.region_name_ccf[~meta.region_name_ccf.isna()], return_counts=True)
    regions = regions[rcnts >= n_ccf_cnt]
    meta = meta[meta.region_name_ccf.isin(regions) & meta.region_id_ccf.isin(SALIENT_REGIONS)]
    r_ccf, rc_ccf = np.unique(meta.region_name_ccf, return_counts=True)
    
    # Load CCF-ME atlas
    me_atlas = load_image(me_atlas_file)
    ccf_atlas = load_image(MASK_CCF25_FILE)
    me2ccf, ccf2me = get_me_ccf_mapper(me2ccf_file)

    t0 = time.time()
    out = []
    for reg in r_ccf:
        print(f'--> {reg}')
        cur_neurons = meta[meta.region_name_ccf == reg]
        # estimate spatial standard deviation
        # for CCF-regions
        centroid = np.array([0, 0, 0])
        variance = 0.0
        count = 0

        # estimate the projection divergence for subregions
        me_regs, me_cnts = np.unique(cur_neurons.region_name_me, return_counts=True)
        me_cnt_mask = me_cnts >= n_me_cnt
        print(reg, me_cnts)
        if me_cnt_mask.sum() <= 1:
            print(f'Insufficient number of subregions within the CCF region: {reg}')
            continue

        # estimate the spatial variance for subregions
        me_variances = []
        for me_reg in me_regs[me_cnt_mask]:
            me_centroid, me_variance, me_count = np.array([0,0,0]), 0.0, 0
            me_neurons = cur_neurons[cur_neurons.region_name_me == me_reg]
            for ifn, fn in enumerate(me_neurons.index):
                neuron_file = os.path.join(swc_dir, f'{fn}.swc')
                # parse the neuron
                dfn = pd.read_csv(neuron_file, index_col=0, sep=' ',
                    names=['t', 'x', 'y', 'z', 'r', 'p'], comment='#'
                )
                # disrecard the soma
                dfn = dfn[dfn.p != -1]
                cur_coords = (dfn[['x', 'y', 'z']] / 1000.).values
                centroid, variance, count = update_spatial_variance(centroid, variance, count, cur_coords)
                me_centroid, me_variance, me_count = update_spatial_variance(
                    me_centroid, me_variance, me_count, cur_coords
                )
                if ifn % 20 == 0:
                    print(f'----> [{me_reg}, {ifn}] {fn}, {variance:.3f}. Time used: {time.time() - t0:.3f} s')
            
            me_variances.append(me_variance)

        # do random parcellation
        

        std = np.sqrt(variance)
        me_std = np.mean(np.sqrt(np.array(me_variances)))
        out.append([reg, std, me_std])
        print(f'<----> {reg}: {std:.3f}, {me_std:.3f}')
        
    out = pd.DataFrame(np.array(out), columns=['region_ccf', 'std_ccf', 'std_me'])
    out.to_csv(out_csv, index=False)
    
        
    
if __name__ == '__main__':
    axon_dir = '/data/lyf/data/fullNeurons/all_neurons_axons/ion_hip_8um'
    meta_file = '../whole-brain_projection/data/meta_hip.csv'
    me_atlas_file = '../../intermediate_data/parc_r671_full_hemi2.nrrd'
    me2ccf_file = '../../intermediate_data/parc_r671_full.nrrd.pkl'
    out_csv = 'temp.csv'
    only_tip = False

    projection_divergence(axon_dir, meta_file, out_csv, me_atlas_file=me_atlas_file, 
        me2ccf_file=me2ccf_file, only_tip=only_tip
    )

