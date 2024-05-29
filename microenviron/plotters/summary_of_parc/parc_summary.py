##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-28
#Description:               
##########################################################
import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pysal.lib as pslib
from esda.moran import Moran

from anatomy.anatomy_config import MASK_CCF25_FILE, SALIENT_REGIONS, \
                                   BSTRUCTS4, BSTRUCTS7, BSTRUCTS13
from file_io import load_image
from math_utils import get_exponent_and_mantissa
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree

sys.path.append('../../')
from config import BS7_COLORS
from parcellation import load_features

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

    def correlation_of_subparcs(self, me_file):
        assert(self.parc_file is not None)
        with open(f'{self.parc_file}.pkl', 'rb') as fp:
            p2ccf = pickle.load(fp)
        # load the microenviron features
        df, fnames = load_features(me_file, feat_type='full')

        ccf2p = {}  # ccf region to subparcellations
        for parc_id, ccf_id in p2ccf.items():
            try:
                ccf2p[ccf_id].append(parc_id)
            except KeyError:
                ccf2p[ccf_id] = [parc_id]

        print('Get the volumes')
        data = []
        nn = 0
        # the volume calculation is time-costly, pre-calculate
        vol_file = './cache/volumes_of_ccf_regions.pkl'
        moran_file = './cache/moranI_of_micro_environ.pkl'
        vol_cached = os.path.exists(vol_file)
        if vol_cached:
            print(f'Loading cached volume file: {vol_file}')
            with open(vol_file, 'rb') as fp:
                vol_dict = pickle.load(fp)
        else:
            vol_dict = {}
        # moran
        moran_cached = os.path.exists(moran_file)
        if moran_cached:
            print(f'Loading cached moran file: {moran_file}')
            with open(moran_file, 'rb') as fp:
                moran_dict = pickle.load(fp)
        else:
            moran_dict = {}
        
        for ccf_id, parc_ids in ccf2p.items():
            # volume 
            if vol_cached:
                vol = vol_dict[ccf_id]
            else:
                cur_mask = self.ccf == ccf_id
                vol = cur_mask.sum() / 40**3
                vol_dict[ccf_id] = vol

            # features of current region
            dfi = df[df.region_id_r671 == ccf_id]
            if dfi.shape[0] < 10:
                fstd = 0
                avgI = 0
            else:
                # variance
                fstd = np.mean(dfi[fnames].std())
                
                if moran_cached:
                    avgI = moran_dict[ccf_id]
                else:
                    # spatial coherence
                    coords = dfi[['soma_x', 'soma_y', 'soma_z']]/40
                    weights = pslib.weights.DistanceBand.from_array(coords.values, threshold=0.5)
                    avgI = []
                    for fn in fnames:
                        moran = Moran(dfi[fn], weights)
                        avgI.append(moran.I)
                    avgI = np.max(avgI)

                    moran_dict[ccf_id] = avgI
            
            num_parc = len(parc_ids)
            data.append((vol, num_parc, fstd, avgI, dfi.shape[0]))
            
            nn += 1
            if nn % 10 == 0:
                print(f'==> Processed {nn} regions')

        # caching the volume dict
        if not vol_cached:
            print(f'Saving the volume file: {vol_file}')
            with open(vol_file, 'wb') as fp:
                pickle.dump(vol_dict, fp)

        if not moran_cached:
            print(f'Saving the moran file: {moran_file}')
            with open(moran_file, 'wb') as fp:
                pickle.dump(moran_dict, fp)

        ####### Volume vs number of regions
        nsub = 'No. of subregions'
        vol_name = r'Volume ($mm^3$)'
        std_name = 'Feature STD'
        moran_name = 'Moran_I'
        nme = 'No. of neurons'
        data = pd.DataFrame(data, columns=(vol_name, nsub, std_name, moran_name, nme))
        
        print('Plotting overall')
        g = sns.regplot(data=data, x=vol_name, y=nsub, scatter_kws={'s':4, 'color':'black'}, 
                        line_kws={'color':'red'})
        r1, p1 = stats.pearsonr(data[vol_name], data[nsub])
        plt.text(0.55, 0.64, r'$R={:.2f}$'.format(r1), transform=g.transAxes)
        e1, m1 = get_exponent_and_mantissa(p1)
        plt.text(0.55, 0.56, r'$P={%.1f}x10^{%d}$' % (m1, e1), transform=g.transAxes)
        plt.ylim(0, data[nsub].max()+1)
        plt.yticks(range(0, data[nsub].max()+1))
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig('volume_vs_nsubparcs_total.png', dpi=300)
        plt.close()

        print('Plotting inset')
        xlim = 3
        data_sub = data[data[vol_name] <= xlim]
        g2 = sns.regplot(data=data_sub, x=vol_name, y=nsub, scatter_kws={'s':4, 'color':'black'}, 
                        line_kws={'color':'red'})
        r2, p2 = stats.pearsonr(data_sub[vol_name], data_sub[nsub])
        plt.text(0.6, 0.9, r'$R={:.2f}$'.format(r2), transform=g2.transAxes)
        e2, m2 = get_exponent_and_mantissa(p2)
        plt.text(0.6, 0.82, r'$P={%.1f}x10^{%d}$' % (m2, e2), transform=g2.transAxes)
        plt.ylim(0, data_sub[nsub].max()+1)
        plt.xlim(0, xlim)
        plt.yticks(range(0, data_sub[nsub].max()+1))
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig('volume_vs_nsubparcs_inset.png', dpi=300)
        plt.close()

        ###### plot the spatial coherent score
        data_salient = data[data[std_name] != 0]

        # scatterplot with histogram at right and bottom
        fig = plt.figure(layout='constrained')
        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        #ax.set(aspect=1)
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
        # scatterplot
        gs = sns.scatterplot(
            data=data_salient, x=moran_name, y=std_name, hue=nsub, size=nsub, ax=ax
        )
        ax.legend(title = '   No. of \nsubregions', labelspacing=0.1, handletextpad=0., 
                   borderpad=0.05, frameon=True, loc='lower right', title_fontsize=15, 
                   fontsize=13, alignment='center', ncols=2)
        # histogram
        def get_values(xy, cs, bins):
            bw = (xy.max() - xy.min()) / bins
            data = []
            xys = []
            for xyi in np.arange(xy.min(), xy.max(), bw):
                xym = (xy >= xyi) & (xy < xyi+bw)
                if xym.sum() < 2: continue
                vs = np.mean(cs[xym])
                xys.append(xyi + bw/2)
                data.append(vs)
            return xys, data

        ax_histx.tick_params(axis='x', labelbottom=False)
        ax_histy.tick_params(axis='y', labelleft=False)
        
        nbins = 20
        ax_histx.plot(*get_values(data_salient[moran_name], data_salient[nsub], nbins), 'o-', c='orchid')
        ax_histy.plot(*get_values(data_salient[std_name], data_salient[nsub], nbins)[::-1], 'o-', c='orchid')

        ax_histx.set_ylabel('No. of\nsubregions', fontsize=13)
        ax_histx.yaxis.set_label_position('right')
        
        ax_histy.set_xlabel('No. of\nsubregions', fontsize=13)

        #plt.subplots_adjust(bottom=0.15)
        plt.savefig('moran_fstd_nsubparcs.png', dpi=300); plt.close() 
        print()
        


if __name__ == '__main__':
    is_ccf = False
    parc_file = '../../intermediate_data/parc_r671_full.nrrd'
    me_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'

    ps = ParcSummary(parc_file, is_ccf=is_ccf)

    if 0:   # region distribution
        ps.region_distributions()

    if 1: 
        ps.correlation_of_subparcs(me_file=me_file)
