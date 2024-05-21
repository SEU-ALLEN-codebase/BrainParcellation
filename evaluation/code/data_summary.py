##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-21
#Description:               
##########################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        bstructs = self.df['bstruct']
        rnames, counts = np.unique(bstructs[bstructs != ''], return_counts=True)
        colors = [self.COLORS[rname] for rname in rnames]
        plt.pie(counts, labels=rnames, colors=colors, autopct='%1.1f%%')
        plt.axis('equal')
        plt.savefig('neuron_distr_among_structures.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    mefile = '../../microenviron/data/mefeatures_100K_with_PCAfeatures3.csv'

    nd = NeuronDistribution(mefile)
    nd.distribution_across_structures()

