##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-21
#Description:               
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import Counter

from anatomy.anatomy_config import SALIENT_REGIONS
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree

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

        plt.pie(counts, labels=bnames, colors=colors, autopct='%1.1f%%', explode=explode)
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
            col1, col2 = 'Brain Structure', 'Num of Neurons'
            df_t = pd.DataFrame(np.array([[k[1], v] for k,v in cnter.items()]), columns=[col1, col2])
            import ipdb; ipdb.set_trace()
            print()
        print()

class QualityEstimation:
    def __init__(self, match_file, gs_file, rec_file):
        self.dfg, self.dfr = self.get_matched(match_file, gs_file, rec_file)
        
        
    def get_matched(self, match_file, gs_file, rec_file):
        dfm = pd.read_csv(match_file, sep=' ', index_col=0)
        dfg = pd.read_csv(gs_file, index_col=0)
        dfr = pd.read_csv(rec_file, index_col=0)

        dfri = dfr.loc[dfm.o_name]
        dfgi = dfg.loc[dfm.index]

        return dfgi, dfri

    def compare_features(self):
        ratios = self.dfr / self.dfg.values
        pf2label = {
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
        
        # select only the target features
        df = ratios[pf2label.keys()].rename(columns=pf2label)
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
        plt.savefig(f'relative_features.png', dpi=300)
        plt.close('all')

        print()
    

if __name__ == '__main__':
    if 1:
        mefile = '../../microenviron/data/mefeatures_100K_with_PCAfeatures3.csv'

        nd = NeuronDistribution(mefile)
        nd.distribution_across_structures()

    if 0:
        match_file = '../data/so_match_table.txt'
        #gs_file = '../data/gf_1876_crop_2um.csv'
        gs_file = '../data/gf_1876_crop_2um_dendrite.csv'
        rec_file = '../../microenviron/data/gf_179k_crop_resampled.csv'

        qe = QualityEstimation(match_file, gs_file, rec_file)
        qe.compare_features()
        

