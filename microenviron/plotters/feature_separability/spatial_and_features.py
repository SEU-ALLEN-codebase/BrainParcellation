##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
import sys
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

from anatomy.anatomy_config import BSTRUCTS13, SALIENT_REGIONS
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree

sys.path.append('../../')
from config import mRMR_f3, mRMR_f3me, standardize_features, load_features

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
                feat = 'Avg. Fragmentation'
            elif feat == 'AverageContraction':
                feat = 'Avg. Straightness'
            elif feat == 'Length':
                feat = 'Total Length'
            #g.axes[-1,i].set_xlabel(feat, fontweight='normal', rotation=25)

        g.despine(bottom=True, left=True)
        plt.subplots_adjust(left=-0.1, bottom=0.05)
        plt.ylabel('Brain area', fontweight='normal')

        plt.savefig(f'areal-feature_distribution.png', dpi=300)
        plt.close('all')
    
    if 1:
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.35)
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
        plt.xticks(rotation=45, rotation_mode='anchor', ha='right', va='top')
        
        cbar = g.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10, direction='in')
        cbar.ax.set_yticks([0,0.2,0.4,0.6,0.8])
        cbar.ax.set_ylabel('Probability', fontsize=10)

        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.savefig('inter-areal_top1_distribution.png', dpi=300)
        plt.close()
        
class SpatialFeatures:
    def __init__(self, feat_file, feat_type='full', min_neurons=10, standardize=True):
        self.df, self.fnames_orig = load_features(feat_file, feat_type=feat_type, standardize=standardize)
        self.min_neurons = min_neurons
        self.feat_type = feat_type
        if self.feat_type == 'full':
            self.fnames = mRMR_f3me
        elif self.feat_type == 'single':
            self.fnames = mRMR_f3
        elif self.feat_type == 'PCA':
            self.fnames = ['pca_feat1', 'pca_feat2', 'pca_feat3']
        else:
            raise NotImplementedError

    def calc_feature_std(self, outfile):
        df = self.df.copy()
        
        STDs = {}
        for irid, rid in enumerate(SALIENT_REGIONS):
            rmask = df.region_id_r671 == rid
            if rmask.sum() < self.min_neurons:
                continue

            df_r = df[rmask][self.fnames]
            std = df_r.std() / df_r.mean()
            #print(df_r.std(), df_r.mean(), std)
            STDs[rid] = std.mean()
            
        with open(outfile, 'wb') as fp:
            pickle.dump(STDs, fp)
            

    def spatial_randomness(self):
        
        # helper functions
        def top1_dist(dfr):
            A = kneighbors_graph(dfr[self.fnames].values, n_neighbors=1, include_self=False,
                                 mode='distance', metric='euclidean', n_jobs=10)
            top1 = A.nonzero()[1]
            src_c = dfr[['soma_x', 'soma_y', 'soma_z']] / 1000.
            dst_c = dfr.iloc[top1][['soma_x', 'soma_y', 'soma_z']] / 1000.
            dists = np.linalg.norm(src_c.values - dst_c.values, axis=1)
            return np.median(dists)


        random.seed(1024)
        np.random.seed(1024)

        df = self.df.copy()
        standardize_features(df, self.fnames)
        
        ratios = []
        for irid, rid in enumerate(SALIENT_REGIONS):
            rmask = df.region_id_r671 == rid
            if rmask.sum() < self.min_neurons:
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
                df_rrng.loc[:, self.fnames] = df_rrng[self.fnames].values[dids]
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

def compare_spatial_statis(params):
    with open(params['single_pkl'], 'rb') as fp:
        sdict = pickle.load(fp)
    
    with open(params['me_pkl'], 'rb') as fp:
        mdict = pickle.load(fp)

    svalues = np.array(list(sdict.values()))
    mvalues = np.array(list(mdict.values()))

    sns.set_theme(style='ticks', font_scale=1.6)
    
    if params['type'] == 'moran':
        deltas = mvalues - svalues
        p_imp = 100.0 * (deltas > 0).sum() / deltas.shape[0]
        g = sns.histplot(x=deltas, fill=True, color='gray', 
                     binrange=(params['xmin'], params['xmax']), 
                     alpha=0.7, bins=50, kde=True, stat='proportion', 
                     line_kws={'alpha':1., 'lw': 2})
        g.lines[0].set_color('black')
    
        plt.axvline(x=0, lw=2, linestyle='--', color='red')
        #plt.annotate(f'Percentage \nimprovement: \n{int(round(p_imp))}%', xy=(0.65,0.92), 
        #             xycoords='axes fraction',
        #             ha='center', va='top', color='red')
        plt.xlim(params['xlim0'], params['xlim1'])
        plt.legend(frameon=False)
    else:
        sns.histplot(x=svalues, fill=True, color='royalblue', binrange=(params['xmin'], params['xmax']), 
                     alpha=0.7, bins=50, kde=True, stat='proportion', label='Single neuron', 
                     line_kws={'alpha':1., 'lw':2})
        sns.histplot(x=mvalues, fill=True, color='indianred', binrange=(params['xmin'], params['xmax']), 
                     alpha=0.7, bins=50, kde=True, stat='proportion', label='Microenvironment', 
                     line_kws={'alpha':1., 'lw':2})
        # customize the legend
        plt.legend().remove()

    plt.xlabel(params['xlabel'])
    plt.ylabel('Percentage of regions')

    plt.subplots_adjust(left=0.18, bottom=0.20)
    plt.savefig(params['figname'], dpi=300)
    plt.close()


if __name__ == '__main__':
    feat_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
    if 1:
        structral_distribution(feat_file)

    if 0:
        # calculate statistics for each type
        standardize = False
        feat_type = 'single'
        outfile = 'std_all_regions_avg3_single.pkl'
        sf = SpatialFeatures(feat_file, feat_type=feat_type, standardize=standardize)
        sf.calc_feature_std(outfile)
        
    if 0:
        # compare and plotting between different feature types
        params1 = {
            'type': 'moran',
            'single_pkl': '../summary_of_parc/cache/moranI_of_singleneuron_avg_top3.pkl', 
            'me_pkl': '../summary_of_parc/cache/moranI_of_micro_environ_avg_top3.pkl',
            'xlabel': u"Δ Spatial coherence\n(Δ Moran's Index)",
            'figname': 'moran_single_vs_me.png',
            'xmin': -0.15,
            'xmax': 0.4,
            'xlim0': -0.13,
            'xlim1': 0.3,
        }
        params2 = {
            'type': 'std',
            'single_pkl': './data/std_all_regions_avg3_single.pkl', 
            'me_pkl': './data/std_all_regions_avg3_me.pkl',
            #'xlabel': u"Coefficient of variance\n(CV=σ/μ)",
            'xlabel': u"Coefficient of variance\n(CV=Standard Deviation / Mean)",
            'figname': 'std_single_vs_me.png',
            'xmin': 0.05,
            'xmax': 0.4,
        }
        compare_spatial_statis(params2)


