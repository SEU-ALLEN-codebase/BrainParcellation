##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-06
#Description:               
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mvlearn.embed import CCA, MCCA
from mvlearn.plotting import crossviews_plot

import sys
sys.path.append('../../')
from config import standardize_features


class MEProjAnalyzer:
    def __init__(self, proj_thresh=1000.):
        self.proj_thresh = proj_thresh

    def load_data(self, me_file, proj_file):
        df_me = pd.read_csv(me_file, index_col=0)
        # standardize
        self.me_feats = [col for col in df_me.columns if col.endswith('_me')]
        self.dend_feats = [col[:-3] for col in df_me.columns if col.endswith('_me')]
        standardize_features(df_me, self.me_feats)
        standardize_features(df_me, self.dend_feats)
        # extract only non-NA values
        df_me = df_me.loc[df_me.index[(df_me[self.me_feats].isna().sum(axis=1) == 0) & 
                (df_me[self.dend_feats].isna().sum(axis=1) == 0)]]

        df_proj = pd.read_csv(proj_file, index_col=0)
        # get the common neurons
        df_proj = df_proj.loc[df_me.index]
        # thresholding regions with small projections
        df_proj[df_proj < self.proj_thresh] = 0
        # remove brain regions without projections
        df_proj = df_proj[df_proj.columns[df_proj.sum(axis=0) > 0]]
        # If remove neurons without salient projections across all regions
        df_proj = df_proj.loc[df_proj.index[(df_proj.sum(axis=1) > 0)]]
        # To log-space
        self.df_proj = np.log(df_proj+1)

        # get the common neurons
        self.df_me = df_me.loc[self.df_proj.index]
        
    def mcca(self, dendrite_type, dataset):

        def pairplot_(df, xy_vars, outfig, dmin1, dmax1, dmin2, dmax2):
            g = sns.pairplot(df, hue='region', 
                        x_vars = xy_vars, 
                        y_vars = xy_vars, 
                        plot_kws={'s':5, 'alpha':0.5, 'edgecolor': None}, 
                        diag_kws={'common_norm': False}
            )
            # customize the xlim
            for ax in g.axes.flatten():
                if ax.get_xlabel() == xy_vars[0]:
                    ax.set_xlim(dmin1, dmax1)
                elif ax.get_xlabel() == xy_vars[1]:
                    ax.set_xlim(dmin2, dmax2)
                if ax.get_ylabel() == xy_vars[0]:
                    ax.set_ylim(dmin1, dmax1)
                elif ax.get_ylabel() == xy_vars[1]:
                    ax.set_ylim(dmin2, dmax2)
            plt.savefig(outfig, dpi=300)
            plt.close()


        # Inititalize CCA
        cca = MCCA(n_components=2, signal_ranks=[2, 2, 2])
        dend2d, me2d, proj2d = cca.fit_transform([self.df_me[self.dend_feats], self.df_me[self.me_feats], self.df_proj])
        # Prepare DataFrame for Seaborn
        df = pd.DataFrame({
            'dend1': dend2d[:, 0],
            'dend2': dend2d[:, 1],
            'me1': me2d[:, 0],
            'me2': me2d[:, 1],
            'proj1': proj2d[:, 0],
            'proj2': proj2d[:, 1],
            'dataset': self.df_me.dataset,
            'region': self.df_me.region_name_r316
        })

        df = df[df.dataset == dataset]
        rs, rcs = np.unique(df.region, return_counts=True)
        krs = rs[rcs > 50]
        df = df[df.region.isin(krs)]

        # unify the min and max of local dendrites
        dmin1, dmax1 = np.percentile(df[['dend1', 'me1']], (1,99))
        dmin2, dmax2 = np.percentile(df[['dend2', 'me2']], (1,99))

        fig_sn_name = f'sn_{dendrite_type}_proj-ccf_{dataset}.png'
        pairplot_(df, ('dend1', 'dend2', 'proj1', 'proj2'), fig_sn_name, dmin1, dmax1, dmin2, dmax2)
        
        fig_me_name = f'me_{dendrite_type}_proj-ccf_{dataset}.png'
        pairplot_(df, ('me1', 'me2', 'proj1', 'proj2'), fig_me_name, dmin1, dmax1, dmin2, dmax2)
        
        
        print()


if __name__ == '__main__':
    dendrite_type = 'dendrites'
    dataset = 'mouselight' # or 'all'
    me_file = f'./data/mefeatures_{dendrite_type}.csv'
    proj_file = '../whole-brain_projection/data/proj.csv'

    mepa = MEProjAnalyzer()
    mepa.load_data(me_file, proj_file)
    mepa.mcca(dendrite_type, dataset)

