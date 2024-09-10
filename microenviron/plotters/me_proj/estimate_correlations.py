##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-06
#Description:               
##########################################################
import os
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mvlearn.embed import CCA, MCCA
from mvlearn.plotting import crossviews_plot

import sys
sys.path.append('../../')
from config import standardize_features

sns.set_theme(style='ticks', font_scale=1.6)


class MEProjAnalyzer:
    def __init__(self, proj_thresh=1000., random_seed=1024):
        self.proj_thresh = proj_thresh
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

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
        
    # The problem is that the dimensionality of projection is too high, correlation analysis resulted 
    # reduced-dimension (n=2) vector could not represent the projection. I verified using PCA on the 
    # projection of HIP neurons, with only <19% for the first two components, 32% for the first five 
    # components
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
        
    def comparative_umap(self, umap_proj=True, dataset='all', nsub=10000):
        import umap
        from scipy.spatial.distance import pdist
        from scipy.stats import linregress
        import statsmodels.api as sm

        #### --- helper functions ------###
        def displot_(pd1, pd2, sub_indices, t, dataset):
            data = pd.DataFrame(np.array((pd1, pd2)).transpose(), columns=(t, 'proj'))
            if sub_indices is not None:
                data = data.iloc[sub_indices]
            g = sns.displot(data=data, x=t, y='proj', pthresh=0.05)
            
            # fitting the lines
            lr = linregress(data[t], data['proj'])
            slope, intercept = lr.slope, lr.intercept
            print(f'{t}: {lr.rvalue:.3f}')
            
            #X_with_constant = sm.add_constant(data[t])
            #ols_model = sm.OLS(data['proj'], X_with_constant).fit()
            #weights = 1 / abs(ols_model.resid)
            #wls_model = sm.WLS(data['proj'], X_with_constant, weights=weights).fit()
            #intercept, slope = wls_model.params
            #print(f'R-squared for {t}: {wls_model.rsquared:.3f}')
            
            lr_fn = np.poly1d([slope, intercept])
            plt.plot(data[t].values, lr_fn(data[t]), '-r')
            plt.text(0.15, 0.8, r'$Coeff={:.3f}$'.format(lr.rvalue), transform=g.ax.transAxes, color='r')
            plt.savefig(f'{t}_{dataset}.png')
            plt.close()
        # --------- End of helper functions ----------


        # umap projection to 2D space
        if umap_proj:
            emb_cache = 'umap_embedding.pkl'
            if os.path.exists(emb_cache):
                with open(emb_cache, 'rb') as fp:
                    emb_me, emb_dend, emb_proj = pickle.load(fp)
            else:
                reducer = umap.UMAP(random_state=self.random_seed)
                emb_me = reducer.fit_transform(self.df_me[self.me_feats])
                emb_dend = reducer.fit_transform(self.df_me[self.dend_feats])
                emb_proj = reducer.fit_transform(self.df_proj)
                # saving
                with open(emb_cache, 'wb') as fp:
                    pickle.dump((emb_me, emb_dend, emb_proj), fp)
        else:
            emb_me = self.df_me[self.me_feats]
            emb_dend = self.df_me[self.dend_feats]
            emb_proj = self.df_proj
        
        # Extract neurons according to the dataset
        if dataset != 'all':
            dataset_ids = np.nonzero(self.df_me.dataset == dataset)[0]
            emb_me = emb_me[dataset_ids]
            emb_dend = emb_dend[dataset_ids]
            emb_proj = emb_proj[dataset_ids]
            print()

        # Extract neurons according to their regions
        region = 'DG-sg'
        if region != 'all':
            region_ids = np.nonzero(self.df_me.region_name_r316 == region)[0]
            emb_me = emb_me[region_ids]
            emb_dend = emb_dend[region_ids]
            emb_proj = emb_proj[region_ids]

        # get the pairwise distance matrices
        pd_me = pdist(emb_me)
        pd_dend = pdist(emb_dend)
        pd_proj = pdist(emb_proj)
        # extract only a subset of the data
        if nsub is not None:
            nsub = min(nsub, len(pd_me))
            sub_indices = random.sample(range(len(pd_me)), nsub)
        displot_(pd_me, pd_proj, sub_indices, 'me', dataset)
        displot_(pd_dend, pd_proj, sub_indices, 'dend', dataset)
        
        print()


if __name__ == '__main__':
    dendrite_type = 'dendrites'
    dataset = 'all' # or 'all'
    me_file = f'./data/mefeatures_{dendrite_type}.csv'
    proj_file = '../whole-brain_projection/data/proj.csv'

    mepa = MEProjAnalyzer()
    mepa.load_data(me_file, proj_file)
    #mepa.mcca(dendrite_type, dataset)
    mepa.comparative_umap(umap_proj=True, dataset=dataset)

