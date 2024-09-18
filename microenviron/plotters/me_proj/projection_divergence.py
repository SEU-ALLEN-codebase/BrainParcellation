##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-13
#Description:               
##########################################################
import os
import sys
import random
import glob
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, BallTree
import matplotlib.pyplot as plt
import seaborn as sns

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
    n_me_cnt = 15
    #regions = ['CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 
    #           'PAR', 'POST', 'PRE', 'ProS', 'SUB']
    regions, rcnts = np.unique(meta.region_name_ccf[~meta.region_name_ccf.isna()], return_counts=True)
    regions = regions[rcnts >= n_ccf_cnt]
    meta = meta[meta.region_name_ccf.isin(regions) & meta.region_id_ccf.isin(SALIENT_REGIONS)]
    r_ccf, rc_ccf = np.unique(meta.region_name_ccf, return_counts=True)
    
    # Load CCF-ME atlas
    me_atlas = load_image(me_atlas_file)
    ccf_atlas = load_image(MASK_CCF25_FILE)
    # We would like to use only the right-hemisphere
    zdim = me_atlas.shape[0]
    zdim2 = zdim // 2
    me_atlas[:zdim2] = 0
    ccf_atlas[:zdim2] = 0
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

        # do random parcellation using Voronoi
        spos = np.floor(cur_neurons[['z', 'y', 'x']] / 25.).astype(int)
        # mirroring
        lr_ids = np.nonzero(spos['z'] < zdim2)[0]
        spos.iloc[lr_ids, 0] = zdim - spos.iloc[lr_ids, 0] - 1
        # random point selection
        reg_id = cur_neurons.region_id_ccf.iloc[0]
        reg_mask = ccf_atlas == reg_id
        # 
        nz_coords = np.array(np.nonzero(reg_mask)).transpose()
        # Get the number of subregions in current CCF region
        nsreg = ccf2me[reg_id]

        # three randomized parcellations
        rnd_variances1, rnd_variances2, rnd_variances3 = [], [], []
        partitions = []
        for i in range(3):
            # randomly select nsreg points
            vcenters = nz_coords[random.sample(range(len(nz_coords)), len(nsreg))]
            # repartition all neurons
            ctree = KDTree(vcenters)
            dmin, imin = ctree.query(spos, k=1)
            partitions.append(imin)
        partitions = np.array(partitions)


        rnd_centroids = np.zeros((1+3, len(nsreg), 3))
        rnd_variances = np.zeros((1+3, len(nsreg)))
        rnd_counts = np.zeros((1+3, len(nsreg)))
        neuron_counts = np.zeros((1+3, len(nsreg)))
        for ifn, fn in enumerate(cur_neurons.index):
            neuron_file = os.path.join(swc_dir, f'{fn}.swc')
            # parse the neuron
            dfn = pd.read_csv(neuron_file, index_col=0, sep=' ',
                names=['t', 'x', 'y', 'z', 'r', 'p'], comment='#'
            )
            # disrecard the soma
            dfn = dfn[(dfn.p != -1) & (dfn.t != 1)]

            if only_tip:
                tip_ids = set(dfn.index) - set(dfn.p)
                dfn = dfn[dfn.index.isin(tip_ids)]

            cur_coords = (dfn[['x', 'y', 'z']] / 1000.).values
            # update the variance for CCF regions
            centroid, variance, count = update_spatial_variance(centroid, variance, count, cur_coords)
            # update for ccf-me subregions
            me_id = int(cur_neurons.loc[fn].region_name_me.split('-R')[-1]) - 1
            rnd_centroids[0,me_id], rnd_variances[0,me_id], rnd_counts[0,me_id] = update_spatial_variance(
                rnd_centroids[0,me_id], rnd_variances[0,me_id], rnd_counts[0,me_id], cur_coords)
            # update for random subregions
            r_id1 = partitions[0,ifn]
            rnd_centroids[1,r_id1], rnd_variances[1,r_id1], rnd_counts[1,r_id1] = update_spatial_variance(
                rnd_centroids[1,r_id1], rnd_variances[1,r_id1], rnd_counts[1,r_id1], cur_coords)
            
            r_id2 = partitions[1,ifn]
            rnd_centroids[2,r_id2], rnd_variances[2,r_id2], rnd_counts[2,r_id2] = update_spatial_variance(
                rnd_centroids[2,r_id2], rnd_variances[2,r_id2], rnd_counts[2,r_id2], cur_coords)

            r_id3 = partitions[2,ifn]
            rnd_centroids[3,r_id3], rnd_variances[3,r_id3], rnd_counts[3,r_id3] = update_spatial_variance(
                rnd_centroids[3,r_id3], rnd_variances[3,r_id3], rnd_counts[3,r_id3], cur_coords)

            # update the neuron counts
            neuron_counts[0,me_id] += 1
            neuron_counts[1, r_id1] += 1
            neuron_counts[2, r_id2] += 1
            neuron_counts[3, r_id3] += 1
            
            if (ifn+1) % 40 == 0:
                print(f'----> [{reg}, {ifn}] {fn}, {variance:.3f}. Time used: {time.time() - t0:.3f} s')

        out_i = [reg]
        out_i.append(np.sqrt(variance))
        for inc in range(neuron_counts.shape[0]):
            neuron_counts_i = neuron_counts[inc]
            rnd_mask = neuron_counts_i >= n_me_cnt
            if rnd_mask.sum() > 0:
                cur_v = np.mean(np.sqrt(rnd_variances[inc][rnd_mask]))
                out_i.append(cur_v)
            else:
                out_i.append(-1)
        out.append([reg, out_i[1], out_i[2], out_i[3], out_i[4], out_i[5]])
        print(neuron_counts)
        print(f'<----> {reg}: {out_i[1]:.3f}, {out_i[2]:.3f}, {out_i[3]:.3f}, {out_i[4]:.3f}, {out_i[5]:.3f}\n')
        
    out = pd.DataFrame(np.array(out), columns=['region_ccf', 'std_ccf', 'std_me', 'std_rnd1', 'std_rnd2', 'std_rnd3'])
    out.to_csv(out_csv, index=False, float_format='%g')
    

def plot_divergence(in_csv, fig_out):
    sns.set_theme(style='ticks', font_scale=2.2)
    df = pd.read_csv(in_csv, index_col=0)
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    xmin = df[['std_ccf', 'std_me']].min(axis=None)
    xmax = df[['std_ccf', 'std_me']].max(axis=None)
    xp = np.arange(xmin, xmax+0.05, 0.05)
    sns.lineplot(x=xp, y=xp, c='red', lw=6, ax=ax, alpha=0.75)
    sns.scatterplot(data=df, x='std_ccf', y='std_me', c='black', ax=ax, s=200, alpha=0.75)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    plt.xlabel('')
    plt.ylabel('')
    ticks = np.arange(2.6, 3.7, 0.2)[1:]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.savefig(fig_out, dpi=300)
    plt.close()
        
    
if __name__ == '__main__':
    axon_dir = '/data/lyf/data/fullNeurons/all_neurons_axons/ion_hip_8um'
    meta_file = '../whole-brain_projection/data/meta_hip.csv'
    me_atlas_file = '../../intermediate_data/parc_r671_full_hemi2.nrrd'
    me2ccf_file = '../../intermediate_data/parc_r671_full.nrrd.pkl'
    out_csv = 'temp.csv'
    only_tip = False

    if 0:
        projection_divergence(axon_dir, meta_file, out_csv, me_atlas_file=me_atlas_file, 
            me2ccf_file=me2ccf_file, only_tip=only_tip
        )

    if 1:
        prefix = 'hip_axon_std_min15'
        plot_divergence(in_csv=f'{prefix}.csv', fig_out=f'{prefix}.png')

