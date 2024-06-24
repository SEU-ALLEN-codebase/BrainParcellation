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
from matplotlib.ticker import MaxNLocator
from scipy import stats

from anatomy.anatomy_config import MASK_CCF25_FILE, SALIENT_REGIONS, \
                                   BSTRUCTS4, BSTRUCTS7, BSTRUCTS13
from file_io import load_image
from math_utils import get_exponent_and_mantissa
from anatomy.anatomy_core import get_struct_from_id_path, parse_ana_tree

sys.path.append('../../')
from config import BS7_COLORS, mRMR_f3me, gini_coeff, moranI_score, load_features


def load_salient_hemisphere(ccf, salient_mask_file=None, zdim2=228):
    if salient_mask_file is None:
        import anatomy
        resource_path = os.path.join(os.path.split(anatomy.__file__)[0], 'resources')
        salient_mask_file = os.path.join(resource_path, 'salient671_binary_mask.nrrd')
    
    smask = load_image(salient_mask_file)
    sccf = ccf.copy()
    sccf[smask == 0] = 0
    sccf[:zdim2] = 0
    return sccf

class ParcSummary:
    def __init__(self, parc_file, is_ccf=False, struct_type=7, flipLR=True):
        print(f'--> Loading the parcellation_file: {parc_file}')
        self.ccf = load_image(MASK_CCF25_FILE)
        self.is_ccf = is_ccf
        if is_ccf:
            self.parc = self.ccf
        else:
            self.parc = load_image(parc_file)
            self.parc_file = parc_file

        self.flipLR = flipLR
        if flipLR:
            self.ccf = load_salient_hemisphere(self.ccf)

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
            subparcs = pd.DataFrame(subparcs, columns=('Region', 'No. of subregions', 'Brain structure'))
            
            sns.boxplot(data=subparcs, x='Brain structure', y='No. of subregions', fill=False, 
                        color='black', order=sorted(BSTRUCTS7.values()), width=0.35)
            plt.yticks(range(0,subparcs['No. of subregions'].max()+1, 2), fontsize=16)
            plt.xticks(sorted(BSTRUCTS7.values()), fontsize=16)
            plt.xlabel('Brain structure', fontsize=18)
            plt.ylabel('No. of subregions', fontsize=18)
            plt.subplots_adjust(bottom=0.15)
            ax = plt.gca()
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
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

    def scatter_hist(self, data, xname, yname, hue_name, figname, log_scale=False):
        # scatterplot with histogram at right and bottom
        fig = plt.figure(layout='constrained')
        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        #ax.set(aspect=1)
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
        # scatterplot
        gs = sns.scatterplot(
            data=data, x=xname, y=yname, hue=hue_name, size=hue_name, ax=ax
        )
        #sns.regplot(
        #    data=data, x=xname, y=yname, ax=ax, scatter=False
        #)
        ax_leg = ax.legend(title =hue_name, labelspacing=0.1, handletextpad=0., 
                   borderpad=0.05, frameon=True, loc='lower right', title_fontsize=15, 
                   fontsize=13, alignment='center', ncols=2)
        ax_leg._legend_box.align = 'center'

        # histogram
        def get_values(xy, cs, bins):
            if log_scale:
                xy = np.log(xy+1e-10)
            bw = (xy.max() - xy.min()) / bins
            data = []
            xys = []
            for xyi in np.arange(xy.min(), xy.max(), bw):
                xym = (xy >= xyi) & (xy < xyi+bw)
                if xym.sum() < 2: continue
                vs = np.mean(cs[xym])
                xys.append(xyi + bw/2)
                data.append(vs)
            if log_scale:
                xys = np.exp(xys)
            return xys, data

        ax_histx.tick_params(axis='x', labelbottom=False)
        ax_histy.tick_params(axis='y', labelleft=False)
        
        nbins = 20
        ax_histx.plot(*get_values(data[xname], data[hue_name], nbins), 'o-', c='orchid')
        ax_histy.plot(*get_values(data[yname], data[hue_name], nbins)[::-1], 'o-', c='orchid')

        ax_histx.set_ylabel(hue_name, fontsize=13)
        ax_histx.yaxis.set_label_position('right')
        
        ax_histy.set_xlabel(hue_name, fontsize=13)
        # force integer scale of ticks
        ax_histx.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax_histy.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

        #plt.subplots_adjust(bottom=0.15)
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.savefig(figname, dpi=300)
        plt.close()

    def load_parc_meta(self):
        assert(self.parc_file is not None)
        with open(f'{self.parc_file}.pkl', 'rb') as fp:
            p2ccf = pickle.load(fp)
        
        ccf2p = {}  # ccf region to subparcellations
        for parc_id, ccf_id in p2ccf.items():
            try:
                ccf2p[ccf_id].append(parc_id)
            except KeyError:
                ccf2p[ccf_id] = [parc_id]

        return p2ccf, ccf2p

    def correlation_of_subparcs(self, me_file):
        # load the microenviron features
        df, fnames = load_features(me_file, feat_type='full', flipLR=self.flipLR)
        p2ccf, ccf2p = self.load_parc_meta()


        print('Get the volumes')
        data = []
        nn = 0
        # the moran calculation is time-costly, pre-calculate
        moran_file = './cache/moranI_of_micro_environ_avg3.pkl'

        # calculate the volumes of all regions
        vol_dict = dict(zip(*np.unique(self.ccf[self.ccf > 0], return_counts=True)))

        # moran
        moran_cached = os.path.exists(moran_file)
        if moran_cached:
            print(f'Loading cached moran file: {moran_file}')
            with open(moran_file, 'rb') as fp:
                moran_dict = pickle.load(fp)
        else:
            moran_dict = {}
        
        for ccf_id, parc_ids in ccf2p.items():
            vol = vol_dict[ccf_id] / 40**3

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
                    avgI = moranI_score(coords.values, dfi[mRMR_f3me].values)
                    moran_dict[ccf_id] = avgI
            
            num_parc = len(parc_ids)
            data.append((vol, num_parc, fstd, avgI, dfi.shape[0]))
            
            nn += 1
            if (nn % 10 == 0) and (not moran_cached):
                print(f'==> Processed {nn} regions')

        # caching the moran index information
        if not moran_cached:
            print(f'Saving the moran file: {moran_file}')
            with open(moran_file, 'wb') as fp:
                pickle.dump(moran_dict, fp)

        ####### Volume vs number of regions
        nsub = 'No. of \nsubregions'
        vol_name = r'Volume ($mm^3$)'
        std_name = 'Feature STD'
        moran_name = 'Moran_I'
        nme = 'No. of neurons'
        data = pd.DataFrame(data, columns=(vol_name, nsub, std_name, moran_name, nme))
        
        if 0:
            # DEPRECATED!
            print('Plotting overall')
            g = sns.regplot(data=data, x=vol_name, y=nsub, scatter_kws={'s':6, 'color':'black'}, 
                            line_kws={'color':'red'})
            r1, p1 = stats.pearsonr(data[vol_name], data[nsub])
            plt.text(0.58, 0.64, r'$R={:.2f}$'.format(r1), transform=g.transAxes)
            e1, m1 = get_exponent_and_mantissa(p1)
            plt.text(0.58, 0.56, r'$P={%.1f}x10^{%d}$' % (m1, e1), transform=g.transAxes)
            plt.ylim(0, data[nsub].max()+1)
            plt.yticks(range(0, data[nsub].max()+1, 2))
            plt.subplots_adjust(left=0.15, bottom=0.15)
            ax = plt.gca()
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            plt.savefig('volume_vs_nsubparcs_total.png', dpi=300)
            plt.close()

            print('Plotting inset')
            xlim = 3
            data_sub = data[data[vol_name] <= xlim]
            g2 = sns.regplot(data=data_sub, x=vol_name, y=nsub, scatter_kws={'s':10, 'color':'black'}, 
                            line_kws={'color':'red'})
            r2, p2 = stats.pearsonr(data_sub[vol_name], data_sub[nsub])
            plt.text(0.45, 0.9, r'$R={:.2f}$'.format(r2), transform=g2.transAxes, fontsize=25)
            e2, m2 = get_exponent_and_mantissa(p2)
            plt.text(0.45, 0.8, r'$P={%.1f}x10^{%d}$' % (m2, e2), transform=g2.transAxes, fontsize=25)
            plt.ylim(0, data_sub[nsub].max()+1)
            plt.xlim(0, xlim)
            plt.xticks(np.arange(0,xlim+0.001, 0.5), fontsize=20)
            plt.xlabel('')
            plt.yticks(range(0, data_sub[nsub].max()+1, 2), fontsize=20)
            plt.ylabel('')
            plt.subplots_adjust(left=0.15, bottom=0.15)
            ax = plt.gca()
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            plt.savefig('volume_vs_nsubparcs_inset.png', dpi=300)
            plt.close()

        ###### plot the spatial coherent score
        data_salient = data[data[std_name] != 0]
        self.scatter_hist(data_salient, moran_name, std_name, nsub, 
                          figname='moran_fstd_nsubparcs.png', log_scale=False)
        ###### volume and number of neurons
        self.scatter_hist(data_salient, vol_name, nme, nsub, 
                          figname='volume_neurons_nsubparcs.png', log_scale=True)
        print()
        
    def volume_statistics(self):
        assert(self.parc_file is not None)
        ccf_vdict = dict(zip(*np.unique(self.ccf[self.ccf > 0], return_counts=True)))
        parc_vdict = dict(zip(*np.unique(self.parc[self.parc > 0], return_counts=True)))
        # transform the voxel-scale volume to physical space volume
        for k, v in ccf_vdict.items(): ccf_vdict[k] /= 40**3
        for k, v in parc_vdict.items(): parc_vdict[k] /= 40**3

        ###### volume distribution
        sns.histplot(x=ccf_vdict.values(), binrange=(0,1.5), bins=50, stat='probability', alpha=0.5, label='CCFv3')
        sns.histplot(x=parc_vdict.values(), binrange=(0,1.5), bins=50, stat='probability', alpha=0.5, label='Ours')
        plt.xlim(0, 1.5)
        plt.xlabel('Volume ($mm^3$)')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('volume_distributions.png', dpi=300)
        plt.close()

        ####### volume of subparcellations 
        # loading the mapping between ccf and our parcellation
        p2ccf, ccf2p = self.load_parc_meta()

        # find out all regions with subregions
        ginis = []
        for ccf_id, parc_ids in ccf2p.items():
            if len(parc_ids) > 1:
                vols = np.array([parc_vdict[parc_id] for parc_id in parc_ids])
                cur_gini = gini_coeff(vols / vols.sum())
                ginis.append(cur_gini)
        
        histplot = sns.histplot(x=ginis, stat='density', binrange=(0, 0.8), bins=25)
        #histogram_data = [(patch.get_height(), patch.get_x(), patch.get_width()) for patch in histplot.patches]
        gmean, gstd = stats.norm.fit(ginis)
        # plot the gaussian plot
        xmin, xmax = plt.xlim()
        xg = np.linspace(xmin, xmax, 100)
        yg = stats.norm.pdf(xg, gmean, gstd)
        plt.plot(xg, yg, 'red', linewidth=2, label=r'$Gaussian (\mu=$'+f'{gmean:.2f}, ' + r'$\sigma=$'+f'{gstd:.2f})')

        plt.xlabel('Gini coefficient of subregion volumes')
        plt.legend()
        plt.tight_layout()
        plt.savefig('gini_of_subregions.png', dpi=300)
        plt.close()
        print()
        
    def eval_parc(self, me_file):
        # load the microenviron features
        df, fnames = load_features(me_file, feat_type='full', flipLR=self.flipLR)
        p2ccf, ccf2p = self.load_parc_meta()

        feats = df[fnames]
        zyx = np.floor(df[['soma_z', 'soma_y', 'soma_x']]).astype(int).values
        ids_in_ccf = self.ccf[zyx[:,0], zyx[:,1], zyx[:,2]]
        ids_in_parc = self.parc[zyx[:,0], zyx[:,1], zyx[:,2]]

        std_ratios = []
        for ccf_id, p_ids in ccf2p.items():
            if len(p_ids) == 1:
                continue

            #if ccf_id == 202:
                #import ipdb; ipdb.set_trace()
            # get the neurons in subparcellations
            ccf_feats = feats[ids_in_ccf == ccf_id]
            p_feats = [feats[ids_in_parc == p_id] for p_id in p_ids]
            # estimate the STD change after subparcellation
            avg_std = []
            for i in range(len(p_feats)):
                p_feat = p_feats[i]
                if len(p_feat) > 2:
                    avg_std.append(p_feat.std().mean())
            avg_std = np.mean(avg_std)
            r_std = avg_std / ccf_feats.std().mean()
            std_ratios.append(r_std)
        
            # estimate the silhoutte score before and after subparcellation
        
        std_ratios = np.array(std_ratios)
        print()
    


if __name__ == '__main__':
    is_ccf = False
    parc_file = '../../intermediate_data/parc_r671_full.nrrd'
    me_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'

    ps = ParcSummary(parc_file, is_ccf=is_ccf)

    if 0:   # region distribution
        ps.region_distributions()

    if 0: 
        ps.correlation_of_subparcs(me_file=me_file)

    if 1:
        ps.volume_statistics()

    if 0:
        ps.eval_parc(me_file=me_file)


