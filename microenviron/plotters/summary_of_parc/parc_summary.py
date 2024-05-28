##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-28
#Description:               
##########################################################
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from anatomy.anatomy_config import MASK_CCF25_FILE, SALIENT_REGIONS, \
                                   BSTRUCTS4, BSTRUCTS7, BSTRUCTS13
from file_io import load_image
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree

sys.path.append('../../')
from config import BS7_COLORS

class ParcSummary:
    def __init__(self, parc_file, is_ccf=False, struct_type=7):
        print(f'--> Loading the parcellation_file: {parc_file}')
        self.ccf = load_image(MASK_CCF25_FILE)
        self.is_ccf = is_ccf
        if is_ccf:
            self.parc = self.ccf
        else:
            self.parc = load_image(parc_file)
            self.parc_file = parc_file

        # brain structures
        if struct_type == 4:
            self.bstructs = BSTRUCTS4
        elif struct_type == 7:
            self.bstructs = BSTRUCTS7
        elif struct_type == 13:
            self.bstructs = BSTRUCTS13
        else:
            raise NotImplementedError
        
        print('--> Get the anatomical tree of ccf atlas')
        self.ana_tree = parse_ana_tree()
        self.r2s = self.regions_to_structures()
        
        # configuring the seaborn theme for following plotting
        sns.set_theme(style='ticks', font_scale=1.5)
    
    def regions_to_structures(self, ignore_zero=True):
        r2s = {}
        bstruct_ids = set(list(self.bstructs))
        for rid in SALIENT_REGIONS:
            # get its structures
            id_path = self.ana_tree[rid]['structure_id_path']
            sid = get_struct_from_id_path(id_path, bstruct_ids)
            if sid == 0 and ignore_zero:
                continue
            sname = self.ana_tree[sid]['acronym']
            r2s[rid] = sname

        if not self.is_ccf:
            # map subparcellations to ccf region, and then to brain structures
            # make sure the correspondence file is located in the same folder of parc file
            with open(f'{self.parc_file}.pkl', 'rb') as fp:
                p2ccf = pickle.load(fp)

            prids = np.unique(self.parc)
            p2s = {}
            s2p = {}
            for prid in prids:
                if prid not in p2ccf: continue
                crid = p2ccf[prid]
                if crid == 997: continue
                sname = r2s[crid]
                p2s[prid] = sname
                try:
                    s2p[crid].append(prid)
                except:
                    s2p[crid] = [prid]

            # plot the region distribution
            subparcs = []
            for idx in r2s.keys():
                subparcs.append([idx, len(s2p[idx]), r2s[idx]])
            subparcs = pd.DataFrame(subparcs, columns=('Region', 'No. of Subregions', 'Brain structure'))
            sns.boxplot(data=subparcs, x='Brain structure', y='No. of Subregions', fill=False, 
                        color='black', order=sorted(BSTRUCTS7.values()), width=0.35)
            plt.savefig('number_of_subregions_distribution.png', dpi=300); plt.close()


            return p2s
        else:               
            return r2s


    def region_distributions(self):
        snames, scnts = np.unique(list(self.r2s.values()), return_counts=True)
        colors = [BS7_COLORS[sname] for sname in snames]
        plt.pie(scnts, labels=snames, colors=colors, autopct='%1.1f%%')
        plt.axis('equal')
        if self.is_ccf:
            rdist_file = 'region_distribution_ccf.png'
        else:
            rdist_file = 'region_distribution_parc_full.png'
        plt.savefig(rdist_file, dpi=300)
        plt.close()
        print()

    def correlation_of_subparcs(self):
        assert(self.parc_file is not None)
        with open(f'{self.parc_file}.pkl', 'rb') as fp:
            p2ccf = pickle.load(fp)

        import ipdb; ipdb.set_trace()
        print()

        


if __name__ == '__main__':
    is_ccf = False
    parc_file = '../../intermediate_data/parc_r671_full.nrrd'

    ps = ParcSummary(parc_file, is_ccf=is_ccf)

    if 0:   # region distribution
        ps.region_distributions()

    if 1: 
        ps.correlation_of_subparcs()
