##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-21
#Description:               
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from adjustText import adjust_text
from collections import Counter
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr

from anatomy.anatomy_config import SALIENT_REGIONS
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree
from ml.feature_processing import standardize_features


PF2LABEL = {
            'AverageBifurcationAngleRemote': 'Bif angle remote',
            'AverageBifurcationAngleLocal': 'Bif angle local',
            'AverageContraction': 'Contraction',
            'AverageFragmentation': 'Avg. Fragmentation',
            'AverageParent-daughterRatio': 'Avg. PD ratio',
            'Bifurcations': 'No. of bifs',
            'Branches': 'No. of branches',
            'HausdorffDimension': 'Hausdorff dimension',
            'MaxBranchOrder': 'Max. branch order',
            'Length': 'Total length',
            'MaxEuclideanDistance': 'Max. Euc distance',
            'MaxPathDistance': 'Max. path distance',
            'Nodes': 'No. of nodes',
            'OverallDepth': 'Overall z span',
            'OverallHeight': 'Overall y span',
            'OverallWidth': 'Overall x span',
            'Tips': 'No. of tips',
        }

REG2STRUCT = {
    'CP': 'STR',
    'SSp-n': 'CTX',
    'AId': 'CTX',
    'CLA': 'CTX',
    'MOs': 'CTX',
    'MOp': 'CTX',
    'SSs': 'CTX',
    'VISp': 'CTX',
    'SSp-bfd': 'CTX',
    'SSp-m': 'CTX',
    'SSp-ul': 'CTX',
    'VPM': 'TH',
    'MG': 'TH',
    'VPL': 'TH',
    'LGd': 'TH',
    'RSPv': 'CTX'
}


class NeuronDistribution:
    BSTRUCTS = {
        688: 'CTX',
        623: 'CNU',
        313: 'MB',
        549: 'TH',
        512: 'CB',
        1065: 'HB',
        1097: 'HY'
    }
    COLORS = {
        'CTX': 'limegreen',
        'CNU': 'darkorange',
        'CB': 'royalblue',
        'TH': 'violet',
        'MB': 'sienna',
        'HY': 'mediumslateblue',
        'HB': 'red'
    }

    def __init__(self, mefile):
        self.df = self.load_data(mefile)
        print(f'#salient neurons: {self.df.shape[0]}')

    def load_data(self, mefile):
        df = pd.read_csv(mefile, index_col=0)
        # only in salient regions
        df = df[df.region_id_r671.isin(SALIENT_REGIONS)]
        # get the brain structures
        structs = []
        bstruct_ids = set(list(self.BSTRUCTS.keys()))
        ana_tree = parse_ana_tree(keyname='id')
        for reg in df.region_id_r671:
            id_path = ana_tree[reg]['structure_id_path']
            sid = get_struct_from_id_path(id_path, bstruct_ids)
            if sid == 0:
                structs.append('')
            else:
                structs.append(ana_tree[sid]['acronym'])
        df['bstruct'] = structs
        return df

    def distribution_across_structures(self):
        sns.set_theme(style="ticks", font_scale=1.2)
        
        ######## overall distribution among brain structures, using pie plot
        bstructs = self.df['bstruct']
        bnames, counts = np.unique(bstructs[bstructs != ''], return_counts=True)
        colors = [self.COLORS[bname] for bname in bnames]
        explode = [0, 0, 0, 0.3, 0.3, 0, 0]

        plt.pie(counts, colors=colors, explode=explode)
        plt.axis('equal')
        plt.savefig('neuron_distr_among_structures.png', dpi=300)
        plt.close()

        ###### For each brain structure ##########
        sns.set_theme(style="ticks", font_scale=1.6)
        if False:
            # distribution for each brain structure
            for bname in bnames:
                dfb = self.df[self.df['bstruct'] == bname]
                rnames, rcnts = np.unique(dfb.region_name_r671, return_counts=True)
                dfc = pd.DataFrame([rnames, rcnts], index=('Region', '#Neurons')).transpose()

                fig, ax = plt.subplots(figsize=(6,6))
                sns.kdeplot(dfc, x='#Neurons', fill=True, alpha=0.2, linewidth=2, color=self.COLORS[bname])
                plt.xlim(0, rcnts.max()*1.2)
                ax.text(0.4, 0.7, f'#Regions (n>0): {len(rcnts)}\n#Regions (n>10): {(rcnts > 10).sum()}\n#Regions (n>100): {(rcnts > 100).sum()}',
                    transform=ax.transAxes)

                plt.title(bname, fontsize=25)
                plt.yticks([])
                ax.set_ylabel('')
                plt.subplots_adjust(bottom=0.15)
                ax.spines['left'].set_linewidth(2)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.savefig(f'{bname}_rcnt_distr.png', dpi=300)
                plt.close()
                
        if True:
            # distributin using box-pots
            df_cur = self.df[['region_name_r671', 'bstruct']][self.df.bstruct.isin(bnames)]
            vcur = [tuple(vi) for vi in df_cur.values]
            cnter = Counter(vcur)
            col1, col2 = 'Brain area', 'Num of neurons'
            df_t = pd.DataFrame(np.array([[k[1], v] for k,v in cnter.items()]), columns=[col1, col2])
            df_t = df_t.astype({'Num of neurons': int})
            g = sns.boxplot(data=df_t, x='Brain area', y='Num of neurons', width=0.35, 
                            color='black', fill=False, order=sorted(np.unique(df_t['Brain area'])))
            plt.yscale('log')
            plt.subplots_adjust(left=0.15, bottom=0.15)
            plt.savefig('neuron_distribution_across_structures.png', dpi=300)
            plt.close()
        print()

class QualityEstimation:
    def __init__(self, match_file, gs_file, rec_file):
        self.dfg, self.dfr, self.dfm = self.get_matched(match_file, gs_file, rec_file)
        
        
    def get_matched(self, match_file, gs_file, rec_file):
        dfm = pd.read_csv(match_file, sep=' ', index_col=0)
        dfg = pd.read_csv(gs_file, index_col=0)
        dfr = pd.read_csv(rec_file, index_col=0)

        dfri = dfr.loc[dfm.o_name]
        dfgi = dfg.loc[dfm.index]

        return dfgi, dfri, dfm

    def plot_relative_boxplots(self, ratios, figout):
        
        
        # In case different neuronal features
        show_keys = list(set(PF2LABEL.keys()) & set(ratios.columns))
        df = ratios[show_keys].rename(columns=PF2LABEL)
        # plot
        sns.set_theme(style='ticks', font_scale=1.6)
        fig = plt.figure(figsize=(12,6))
        rname = 'Relative to manual'
        df = df.stack().reset_index().rename(columns={'level_0': 'neuron', 'level_1': 'Feature', 0: rname})
        sns.boxplot(data=df, x='Feature', y=rname, hue='Feature')
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2)

        axes = plt.gca()
        #axes.set_title(pf, fontsize=font)
        #axes.text(0,2,feature,va='top',ha='center',fontsize=font)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_linewidth(2)
        axes.spines['left'].set_linewidth(2)
        axes.xaxis.set_tick_params(width=2, direction='out')
        axes.yaxis.set_tick_params(width=2, direction='out')
        plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')


        fig.subplots_adjust(left=0.16, bottom=0.38)
        plt.ylim(0., 2.0)
        plt.xlabel('Morphological feature', fontsize=24)
        plt.ylabel('Relative to manual', fontsize=24)
        plt.savefig(figout, dpi=300)
        plt.close('all')


    def compare_features(self):
        ratios = self.dfr / self.dfg.values
        self.plot_relative_boxplots(ratios, f'relative_features.png')
    
    
    def compare_features_recons_me_gs(self, me_file, region_file):
        # load the region file for gold standard
        regions = pd.read_csv(region_file, index_col=0)

        # load the me_features
        dfme_orig = pd.read_csv(me_file, index_col=0)
        # we rename the columns
        fnames_me = [col for col in dfme_orig.columns if col.endswith('_me')]
        # rename the "X_me" to "X" to facilitate subsequent comparison
        dfme = dfme_orig[fnames_me].copy()
        fname_mapper = {}
        for mf in fnames_me:
            fname_mapper[mf] = mf[:-3]
        dfme.rename(columns=fname_mapper, inplace=True)
        # common features
        common_feats = list(set(self.dfg.columns) & set(dfme.columns))

        # remove neurons not contained in the ME dataset
        in_me = self.dfr.index.isin(dfme.index)
        dfr = self.dfr[in_me][common_feats]
        dfg = self.dfg[in_me][common_feats]
        dfme = dfme[dfme.index.isin(dfr.index)][common_feats]

        # reconstruction vs gold standard
        ratios_rg = dfr / dfg.values
        ratios_mg = dfme / dfg.values

        # region-wise comparison
        # compare with me features

    def compute_distances(self, df, class_column='region', reduce_method='median'):
        classes = df[class_column].unique()
        # initialize the distance matrix
        dmat = np.zeros((len(classes), len(classes)))
        print(classes)
        intra_dists = []
        # Compute In-Class Distances
        dist_metric = 'l2'

        if reduce_method == 'median':
            freduce = np.median
        elif reduce_method == 'mean':
            freduce = np.mean
        else:
            raise NotImplementedError

        for icls, cls in enumerate(classes):
            cls_data = df[df[class_column] == cls].drop(columns=[class_column]).values
            if len(cls_data) < 2:
                continue  # Skip classes with less than 2 samples
            distances = pairwise_distances(cls_data, metric=dist_metric)
            triu_indices = np.triu_indices_from(distances, k=1)
            dmat[icls,icls] = freduce(distances[triu_indices])
            intra_dists.extend(distances[triu_indices].tolist())
            #print(np.median(distances))
        
        # Compute Inter-Class Distances
        inter_dists = []
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes[i+1:]):
                j2 = j + i + 1
                cls1_data = df[df[class_column] == cls1].drop(columns=[class_column]).values
                cls2_data = df[df[class_column] == cls2].drop(columns=[class_column]).values
                distances = pairwise_distances(cls1_data, cls2_data, metric=dist_metric)
                dij = freduce(distances)
                dmat[i,j2] = dij
                dmat[j2,i] = dij
                inter_dists.extend(distances.flatten().tolist())
        
        dmat = pd.DataFrame(dmat, index=classes, columns=classes)

        return dmat, intra_dists, inter_dists

    def compare_regional_similarity(self, region_file):
        # load the region file for gold standard
        regions = pd.read_csv(region_file, index_col=0)


        # relative feature values
        ratios = self.dfr.values / self.dfg
        # including only meaningful features
        ratios = ratios[PF2LABEL.keys()]
        # get the brain region
        com_neurons = ratios.index[ratios.index.isin(regions.index)]
        ratios['region'] = ''
        ratios.loc[com_neurons, 'region'] = regions.loc[com_neurons].values

        # remove regions with only limited neurons
        min_neurons = 15
        regs, rcnts = np.unique(ratios.region, return_counts=True)
        regs_k = regs[rcnts >= min_neurons]
        ratios = ratios[ratios.region.isin(regs_k)]

        # estimate the intra-region distance and inter-region distance
        #dmat, d_intra, d_inter = self.compute_distances(ratios, class_column='region')
        #sns.clustermap(dmat, cmap='seismic'); plt.savefig('tmp.png', dpi=300); plt.close()
        
        # Standardize features
        sns.set_theme(style='ticks', font_scale=1.8)
        dfg = self.dfg.loc[ratios.index, ratios.columns[:-1]]
        dfg = standardize_features(dfg, dfg.columns, inplace=False)
        dfg['region'] = ratios['region']
        dmat_g, d_intra_g, d_inter_g = self.compute_distances(dfg, class_column='region', reduce_method='median')
        
        BSTRUCTS = np.unique(list(REG2STRUCT.values()))
        lut_row = {bs:plt.cm.rainbow(each, bytes=False) 
                   for bs, each in zip(BSTRUCTS, np.linspace(0, 1, len(BSTRUCTS)))}
        row_colors_g = [lut_row[REG2STRUCT[reg]] for reg in dmat_g.index]
        g_g = sns.clustermap(dmat_g, cmap='seismic', row_colors=row_colors_g)
        plt.setp(g_g.ax_heatmap.get_xticklabels(), rotation=45, rotation_mode='anchor', ha="right", va='top')
        plt.savefig('manual_regional_heatmap.png', dpi=300); plt.close()
        

        # For the reconstructions
        sns.set_theme(style='ticks', font_scale=1.2)
        plt.figure(figsize=(6,6))
        dfr = self.dfr.loc[self.dfm.loc[dfg.index].o_name]
        dfr = standardize_features(dfr, dfr.columns, inplace=False)
        dfr['region'] = dfg['region'].values
        dmat_r, d_intra_r, d_inter_r = self.compute_distances(dfr, class_column='region', reduce_method='median')
        dmat_r_reordered = dmat_r.iloc[ g_g.dendrogram_row.reordered_ind,  g_g.dendrogram_col.reordered_ind]
        g_r = sns.heatmap(dmat_r_reordered, cmap='seismic', vmin=4.5, vmax=9.,
                          cbar_kws={'aspect': 5, 'fraction': 0.06})
        plt.setp(g_r.get_xticklabels(), rotation=45, rotation_mode='anchor', ha="right", va='top')
        #plt.axis('equal')
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig('auto_regional_heatmap.png', dpi=300); plt.close()

        # estimate the similarity of two distance matrix
        sns.set_theme(style='ticks', font_scale=1.6)
        plt.figure(figsize=(6,6))
        triu_indices = np.triu_indices_from(dmat_g, k=1)
        gvs = dmat_g.values[triu_indices]
        rvs = dmat_r.values[triu_indices]
        df_rgvs = pd.DataFrame(np.array([gvs, rvs]).transpose(), columns=('Auto', 'Manual'))
        ax = sns.regplot(df_rgvs, x='Auto', y='Manual', 
                    scatter_kws={'s': 15, 'color': 'black', 'alpha':0.8},
                    line_kws={'color':'magenta', 'alpha': 0.75}
                    )
        presult = pearsonr(gvs, rvs)
        ax.text(.1, .85, f'Correlation={presult.statistic:.2f}',
                    fontsize=20, transform=ax.transAxes, color='black')
        plt.savefig('auto_vs_manual_distances.png', dpi=300); plt.close()
        #import ipdb; ipdb.set_trace()
        print()



if __name__ == '__main__':
    mefile = '../../microenviron/data/mefeatures_100K_with_PCAfeatures3.csv'
    
    if 0:
        nd = NeuronDistribution(mefile)
        nd.distribution_across_structures()

    if 1:
        match_file = '../data/so_match_table.txt'
        #gs_file = '../data/gf_1876_crop_2um.csv'
        gs_file = '../data/gf_1876_crop_2um_dendrite.csv'
        rec_file = '../../microenviron/data/gf_179k_crop_resampled.csv'
        region_file = '../data/1876_soma_region.csv'

        qe = QualityEstimation(match_file, gs_file, rec_file)
        #qe.compare_features_recons_me_gs(mefile, region_file)
        qe.compare_regional_similarity(region_file)

