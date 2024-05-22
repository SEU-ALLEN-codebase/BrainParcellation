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
            #sys.exit()
        print()

class QualityEstimation:
    def __init__(self, match_file, gs_file, rec_file):
        self.dfg, self.dfr = self.get_matched(match_file, gs_file, rec_file)
        
        
    def get_matched(self, match_file, gs_file, rec_file):
        import ipdb; ipdb.set_trace()
        dfm = pd.read_csv(match_file, sep=' ')
        dfg = pd.read_csv(gs_file)
        dfr = pd.read_csv(rec_file, index_col=0)
    

if __name__ == '__main__':
    if 0:
        mefile = '../../microenviron/data/mefeatures_100K_with_PCAfeatures3.csv'

        nd = NeuronDistribution(mefile)
        nd.distribution_across_structures()

    if 1:
        match_file = '../evaluation/data/so_match_table.txt'
        gs_file = '../evaluation/data/gf_1876_crop_2um.csv'
        rec_file = '../../microenviron/data/gf_179k_crop_resampled.csv'

        qe = QualityEstimation(match_file, gs_file, rec_file)
        
        

