##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

from anatomy.anatomy_config import BSTRUCTS13, SALIENT_REGIONS
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree

sys.path.append('../../')
from config import mRMR_f3, mRMR_f3me, standardize_features

def regions_to_structs(regids, ana_tree):
    bstructs = BSTRUCTS13.keys()
    strs = []
    for regid in regids:
        if regid == 0:
            strs.append('')
            continue
        
        id_path = ana_tree[regid]['structure_id_path']
        sid = get_struct_from_id_path(id_path, bstructs)
        if sid == 0:
            strs.append('')
        else:
            strs.append(BSTRUCTS13[sid])
    return strs

def structral_distribution(feat_file):
    id_key = 'region_id_r671'
    df = pd.read_csv(feat_file, index_col=0)

    ana_tree = parse_ana_tree()
    strs13 = np.array(regions_to_structs(df[id_key], ana_tree))
    strs13m = strs13 != ''
    strs13 = strs13[strs13m]
    df = df[strs13m]
    
    # get only regions with correct brain structures
    uniq_strs13 = np.unique(strs13)
    df_f = df[mRMR_f3me]
    standardize_features(df_f, mRMR_f3me)

    
    if 1:
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=3.2)
        # Part I: ridgeplot showing feature distribution
        # reform the dataset
        dfs = df_f.stack().reset_index(1)
        nf = len(mRMR_f3me)
        structs = [s for s in strs13 for i in range(nf)]
        dfs = dfs.assign(struct=structs)
        # rename the dataframe
        dfs.rename(mapper={'level_1': 'feature', 0: 'value'}, axis=1, inplace=True)

        # Initialize the facegrid
        hspace = -0.65
        height = 1
        g = sns.FacetGrid(dfs, row='struct', col='feature', hue='feature', aspect=6, 
                          height=height, row_order=uniq_strs13)
        
        # Draw the densities in a few steps
        print('Draw kdeplot...')
        bw_adjust = 1.
        xmin, xmax = -5, 5
        g.map(sns.kdeplot, 'value',
              bw_adjust=bw_adjust, clip=(xmin,xmax),
              fill=True, alpha=0.3, linewidth=1.5)
        g.map(sns.kdeplot, 'value', clip=(xmin, xmax), lw=3, bw_adjust=bw_adjust, alpha=0.7)

        # passing color=None to refline() uses the hue mapping
        #g.refline(y=0, linewidth=2, linestyle="-", color='k', clip_on=False)
        #g.map(plt.axhline, y=0, linewidth=2, linestyle="-", color='k', clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def ylabel(x, color, label):
            if label == 'Length_me':
                ax = plt.gca()
                ax.text(0.2, (1+hspace)*0.4, x.iloc[0],# fontweight="bold", 
                        color='k',
                        ha="left", va="center", transform=ax.transAxes,
                        fontweight='normal')

        g.map(ylabel, 'struct')

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=hspace, wspace=-0.55)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel='')
        #g.set(xticks=np.arange(-.5,1+0.5,0.5))
        #g.set_xticklabels(np.arange(-.5,1.5,0.5), fontsize=fontsize-5, fontweight='normal')
        g.set(xticks=[], xlabel='')

        #g.add_legend()
        #sns.move_legend(g, 'lower center', ncols=5)
        #plt.setp(g._legend.get_title(), fontsize=0, fontweight='bold')
        #plt.setp(g._legend.get_texts(), fontsize=fontsize+5, fontweight='bold')
        for i, feat in enumerate(mRMR_f3):
            if feat == 'AverageFragmentation':
                feat = 'AvgFragmentation'
            elif feat == 'AverageContraction':
                feat = 'Straightness'
            g.axes[-1,i].set_xlabel(feat, fontweight='normal', rotation=0)

        g.despine(bottom=True, left=True)
        plt.subplots_adjust(left=-0.1, bottom=0.05)
        plt.ylabel('Brain area', fontweight='normal')

        plt.savefig(f'areal-feature_distribution.png', dpi=300)
        plt.close('all')
    
    if 1:
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.4)
        # Part II: Pairwise similarity
        df_f = df_f.assign(struct=strs13)
        A = kneighbors_graph(df_f[mRMR_f3me].values, n_neighbors=1, include_self=False, 
                             mode='distance', metric='euclidean', n_jobs=10)
        top1 = A.nonzero()[1]
        smap = dict(zip(uniq_strs13, range(len(uniq_strs13))))
        df2 = df_f.copy()
        df2['struct'] = df2['struct'].replace(smap)
        pairs = np.array([df2.struct.values, df2.iloc[top1].struct.values])
        pmat = np.zeros((len(uniq_strs13), len(uniq_strs13)))
        pij, pcnts = np.unique(pairs.transpose(), axis=0, return_counts=True)
        pmat[pij[:,0], pij[:,1]] = pcnts
        # normalize
        pmat /= pmat.sum(axis=1).reshape(-1,1)
        pmat /= pmat.sum(axis=0)
        pmat = pd.DataFrame(pmat, columns=uniq_strs13, index=uniq_strs13)
        g = sns.heatmap(pmat, cmap='Reds', square=True,
                        cbar_kws={'aspect':6, 'shrink':0.3, 'pad':0.015})
        g.tick_params(axis='both', pad=-3)
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10, direction='in')
        cbar.ax.set_yticks([0,0.2,0.4,0.6,0.8])
        cbar.ax.set_ylabel('Probability', fontsize=10)

        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.savefig('inter-areal_top1_distribution.png', dpi=300)
        plt.close()
        

def spatial_randomness(feat_file):
    
    # helper functions
    def top1_dist(dfr):
        A = kneighbors_graph(dfr[mRMR_f3me].values, n_neighbors=1, include_self=False,
                             mode='distance', metric='euclidean', n_jobs=10)
        top1 = A.nonzero()[1]
        src_c = dfr[['soma_x', 'soma_y', 'soma_z']] / 1000.
        dst_c = dfr.iloc[top1][['soma_x', 'soma_y', 'soma_z']] / 1000.
        dists = np.linalg.norm(src_c.values - dst_c.values, axis=1)
        return np.median(dists)


    random.seed(1024)
    np.random.seed(1024)

    df = pd.read_csv(feat_file, index_col=0)
    standardize_features(df, mRMR_f3me)
    
    ratios = []
    for irid, rid in enumerate(SALIENT_REGIONS):
        rmask = df.region_id_r671 == rid
        if rmask.sum() < 10:
            continue
        df_r = df[rmask]
        
        # estimate top1 distance
        mdist1 = top1_dist(df_r)

        # Randomly shuffle the data
        mdists2 = []
        for i in range(5):
            df_rrng = df_r.copy()
            dids = np.arange(df_r.shape[0])
            random.shuffle(dids)
            df_rrng.loc[:, mRMR_f3me] = df_rrng[mRMR_f3me].values[dids]
            mdists2.append(top1_dist(df_rrng))
        mdist2 = np.median(mdists2)

        ratio = mdist1/mdist2
        ratios.append(ratio)
        if ratio > 1:
            print(df_r.region_name_r671.iloc[0], ratio)

        if len(ratios) % 10 == 0:
            print(irid)
            #break

    ratios = np.array(ratios)
    print(f'Mean and median ratio: {ratios.mean()}, {np.median(ratios)}')

    fig = plt.figure(figsize=(6,6))
    sns.set_theme(style='ticks', font_scale=2.1)
    dfr = pd.DataFrame(ratios, columns=['ratio'])
    sns.histplot(x=dfr['ratio'], fill=True, color='mediumpurple', bins=100)
    
    plt.xlim(0, dfr['ratio'].max())
    plt.xticks([0,1,2,3,4,5,6])
    plt.xlabel('Spatial randomness')
    plt.axvline(x=1, linewidth=2, linestyle='--', color='red')

    ax = plt.gca()
    lw = 2.5
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['top'].set_linewidth(lw)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig('non_randomness.png', dpi=300); plt.close()


if __name__ == '__main__':
    feat_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
    if 1:
        structral_distribution(feat_file)

    if 0:
        spatial_randomness(feat_file)


