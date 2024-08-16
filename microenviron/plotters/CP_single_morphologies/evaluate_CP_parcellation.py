##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-02
#Description:               
##########################################################
import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm
import seaborn as sns

from swc_handler import get_soma_from_swc
from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from projection.projection import Projection
from config_CP import SUBREGIONS2COMMU, COMMU2SUBREGIONS, CCF_ID_CP

sns.set_theme(style='ticks', font_scale=1.8)
_COMM_COLORS = {
    'CP.r': 'red',
    'CP.ri': 'orange',
    'CP.i': 'yellow',
    'CP.ic': 'green',
    'CP.c': 'cyan'
}
_PARC_COLORS = {
    1: 'red',
    2: 'orange',
    3: 'yellow',
    4: 'green',
    5: 'cyan',
    6: 'blue',
    7: 'purple',
    8: 'sienna',
    9: 'gold',
    10: 'lime',
    11: 'deepskyblue',
    12: 'violet',
    13: 'dimgray'
}

class EvalParcellation:
    def __init__(self, cp_parc_file):
        self.load_subregions(cp_parc_file)

    def load_subregions(self, cp_parc_file, parc_file=None):
        self.cp_parc = load_image(cp_parc_file)
        self.ccf_atlas = load_image(MASK_CCF25_FILE)
        self.df_slocs = self.get_soma_locs(axon_dir)

    def get_soma_locs(self, axon_dir, flipLR=True):
        slocs = []
        fnames = []
        for axon_file in glob.glob(os.path.join(axon_dir, '*.swc')):
            fname = os.path.split(axon_file)[-1][:-4]
            sline = get_soma_from_swc(axon_file)
            sloc = list(map(float, sline[2:5]))
            slocs.append(sloc)
            fnames.append(fname)

        slocs = np.array(slocs) # xyz
        # to 25um space
        slocs = np.floor(slocs / 25.).astype(int)[:,::-1]

        # flip left to right
        if flipLR:
            zdim = self.cp_parc.shape[0]
            left_mask = slocs[:,0] < zdim/2
            lm_nz = left_mask.nonzero()[0]
            slocs[lm_nz, 0] = zdim - slocs[lm_nz, 0]

        df_slocs = pd.DataFrame(slocs, columns=('z', 'y', 'x'))
        df_slocs.index = fnames
        return df_slocs
        

    def to_proj(self, proj_csv):
        projs = pd.read_csv(proj_csv, index_col=0)
        projs[projs < 1000] = 0
        log_projs = np.log(projs+1)
        
        zs, ys, xs = self.df_slocs.values.transpose()
        sregs = self.cp_parc[zs, ys, xs]
        # make sure the are in the same order
        log_projs = log_projs.loc[self.df_slocs.index]

        # remove zeroing neurons
        nzm = sregs != 0
        log_projs = log_projs[nzm]
        zs = zs[nzm]
        ys = ys[nzm]
        xs = xs[nzm]
        sregs = sregs[nzm]
        
        normalize = True
        if normalize:
            log_projs = log_projs / log_projs.sum(axis=1).values.reshape(-1,1)
        # remove zero columns
        log_projs = log_projs[log_projs.columns[log_projs.sum() != 0]]

        debug = False
        if debug:
            # rename to region:
            # plot the map
            ana_tree = parse_ana_tree()
            # regid to regname
            reg_dict = {}
            for regid in projs.columns:
                regid_int = int(regid)
                if regid > 0:
                    hstr = 'ipsi-'
                    rname = ana_tree[regid_int]['acronym']
                else:
                    hstr = 'contra-'
                    rname = ana_tree[-regid_int]['acronym']
                reg_dict[regid] = hstr + rname
            # 
            log_projs_rname = log_projs.rename(columns=reg_dict)
            log_projs_rname2 = log_projs_rname.copy()
            log_projs_rname2.loc[:, 'ipsi-CP'] = 0
            pregs_t1 = dict(zip(*np.unique(log_projs_rname2.columns[np.argmax(log_projs_rname2, axis=1)], return_counts=True)))
        
        # aggregate informations
        corrs = log_projs.transpose().corr()
        nregs = sregs.max()
        mcorr = np.ones((nregs, nregs))
        for ireg in range(1, nregs+1):
            iids = np.nonzero(sregs == ireg)[0]
            print(f'# of neurons in region {ireg-1}: {iids.shape[0]}')
            for jreg in range(ireg, nregs+1):
                jids = np.nonzero(sregs == jreg)[0]
                msub = corrs.iloc[iids, jids]
                if ireg != jreg:
                    avg_corr = msub.mean(axis=None)
                else:
                    pids1, pids2 = np.triu_indices_from(msub, k=1)
                    avg_corr = msub.values[pids1, pids2].mean()
                mcorr[ireg-1, jreg-1] = avg_corr
                mcorr[jreg-1, ireg-1] = avg_corr
        
        # Intra-/Inter-subregional similarity       
        names = [f'R{i+1}' for i in range(mcorr.shape[0])]
        df_mcorr = pd.DataFrame(mcorr, index=names, columns=names)
        sns.clustermap(df_mcorr, cmap='hot_r')
        plt.savefig('subregion_vs_projection.png', dpi=300); plt.close()
        
        #import ipdb; ipdb.set_trace()
        print()
        

    def projected_subregions(self, proj_csv, file_me2ccf, meta_table):
        projs = pd.read_csv(proj_csv, index_col=0)
        projs.columns = projs.columns.astype(int)
        projs[projs < 100] = 0
        log_projs = np.log(projs+1)
        # load the meta information for cp neurons
        meta = pd.read_excel(meta_table, index_col=0)
        cp_neurons = meta[meta['Projection class'].isin(['CP_SNr', 'CP_GPe', 'CP_others'])]
        
        zs, ys, xs = self.df_slocs.values.transpose()
        sregs = self.cp_parc[zs, ys, xs]
        # make sure the are in the same order
        log_projs = log_projs.loc[self.df_slocs.index]

        # remove zeroing neurons
        nzm = sregs != 0
        log_projs = log_projs[nzm]
        zs = zs[nzm]
        ys = ys[nzm]
        xs = xs[nzm]
        sregs = sregs[nzm]
        
        normalize = False
        if normalize:
            log_projs = log_projs / log_projs.sum(axis=1).values.reshape(-1,1)
        # remove zero columns
        #log_projs = log_projs[log_projs.columns[log_projs.sum() != 0]]

        # identify 
        with open(file_me2ccf, 'rb') as fp:
            me2ccf = pickle.load(fp)
        # ccf2me
        ccf2me = {}
        for k,v in me2ccf.items():
            if v in ccf2me:
                ccf2me[v].append(k)
            else:
                ccf2me[v] = [k]

        # GPe: 1022, SNr: 381
        gpe = cp_neurons[cp_neurons['Projection class'] == 'CP_GPe']
        snr = cp_neurons[cp_neurons['Projection class'] == 'CP_SNr']
        others = cp_neurons[cp_neurons['Projection class'] == 'CP_others']

        ptype_params = {
            'GPe': {
                'neurons': gpe,
                'xlabel': 'Subregions of GPe',
                'ylabel': 'GPe-projecting CP neurons',
                'figname': 'GPe-projecting_subregions.png',
                'ccf_id': 1022
            },
            'SNr': {
                'neurons': snr,
                'xlabel': 'Subregions of SNr',
                'ylabel': 'SNr-projecting CP neurons',
                'figname': 'SNr-projecting_subregions.png',
                'ccf_id': 381
            },
        }

        for tname, ptype in ptype_params.items():
            # projections for each ptype
            xlabel = ptype['xlabel']
            ylabel = ptype['ylabel']
            figname = ptype['figname']
            cur_neurons = ptype['neurons']
            
            # plotting the projection heatmap of different ptypes
            cur_projs = log_projs[log_projs.index.isin(cur_neurons.index)]
            sub_ids = ccf2me[ptype['ccf_id']]
            sub_subregions = sub_ids + [-idx for idx in sub_ids]
            sub_projs = cur_projs[sub_subregions]
            sub_projs = sub_projs[sub_projs.columns[sub_projs.sum() != 0]]
            
            rndict = {}
            min_id = np.min(sub_ids)
            for sub_id in sub_ids:
                if sub_id > 0:
                    rndict[sub_id] = tname + str(sub_id - min_id + 1)
                else:
                    rndict[sub_id] = tname + str(sub_id - min_id + 1) + '-contra'
            sub_projs.rename(columns=rndict, inplace=True)
            
            # evaluate the relationship between projection patterns and CP subregions
            cur_zyx = self.df_slocs.loc[sub_projs.index]
            cp_subregions = self.cp_parc[cur_zyx.z, cur_zyx.y, cur_zyx.x] - min(ccf2me[CCF_ID_CP]) + 1
            # coloring by cp subregions
            #lut_cp = {lab:plt.cm.rainbow(each)[:3] 
            #          for lab,each in zip(range(1, len(ccf2me[CCF_ID_CP])+1), np.linspace(0,1,len(ccf2me[CCF_ID_CP])))}
            #col_colors1 = pd.Series(cp_subregions, name='Subregions\nof CP').map(lut_cp).values
            
            # coloring by community
            cp_comms = np.array([SUBREGIONS2COMMU[i] for i in cp_subregions])
            lut_cp = {lab:plt.cm.rainbow(each)[:3]
                      for lab,each in zip(COMMU2SUBREGIONS.keys(), np.linspace(0,1,len(COMMU2SUBREGIONS)))}
            col_colors1 = pd.Series(cp_comms, name='Subregions\nof CP').map(lut_cp).values

            # plot
            sub_projs_t = sub_projs.transpose()
            g = sns.clustermap(sub_projs_t, cmap='hot_r',
                               cbar_pos=(0.8,0.1,0.03,0.2), 
                               figsize=(10,8)
                               )
            g.ax_heatmap.tick_params(axis='x', bottom=False, labelbottom=False)
            g.ax_heatmap.tick_params(axis='y', left=True, labelleft=True, right=False, 
                                     labelright=False)
            g.ax_heatmap.set_ylabel(xlabel, fontsize=26)
            g.ax_heatmap.yaxis.set_label_position("left")
            for label in g.ax_heatmap.get_yticklabels():
                label.set_rotation(0)  # Rotate y-tick labels by 90 degrees

            g.ax_heatmap.set_xlabel(ylabel, labelpad=6, fontsize=26)
            # spines
            g.ax_heatmap.spines['left'].set_visible(True)
            g.ax_heatmap.spines['left'].set_linewidth(1)
            g.ax_heatmap.spines['right'].set_visible(True)
            g.ax_heatmap.spines['right'].set_linewidth(1)
            g.ax_heatmap.spines['top'].set_visible(True)
            g.ax_heatmap.spines['top'].set_linewidth(1)
            g.ax_heatmap.spines['bottom'].set_visible(True)
            g.ax_heatmap.spines['bottom'].set_linewidth(1)
            # colorbar configuration
            #g.cax.tick_params(direction='in')
            cyticks = np.arange(0,sub_projs_t.max(axis=None),2)
            g.cax.set_yticks(cyticks, cyticks)
            g.cax.set_ylabel(r'$\ln(L+1)$')
            #g.cax.yaxis.set_label_position("left")
            # hide the dendrogram
            g.ax_col_dendrogram.set_visible(False)
            g.ax_row_dendrogram.set_visible(False)

            #plt.subplots_adjust(bottom=0.08)
            plt.savefig(figname, dpi=300); plt.close()


            ################# Composition of CP regions to each target subregions
            #reordered_ind_col = g.dendrogram_col.reordered_ind
            #reordered_ind_row = g.dendrogram_row.reordered_ind
            for tsreg in sub_projs_t.index:
                # projection in current region
                cur_cp_sregs = cp_subregions[sub_projs_t.loc[tsreg] > 0]
                cur_cp_comms = cp_comms[sub_projs_t.loc[tsreg] > 0]
                # pie chart for comm
                snames_comm, scnts_comm = np.unique(cur_cp_comms, return_counts=True)
                colors_comm = [_COMM_COLORS[comm] for comm in snames_comm]
                fig_pie = plt.pie(scnts_comm, colors=colors_comm)
                
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(f'{tname}_{tsreg}_cp_subregion_distr.png', dpi=300)
                plt.close()
                
            ################ projection clusters vs CP regions ##################
            #import ipdb; ipdb.set_trace()
            print()
            
            

            # ------------------#
            if False:
                # estimate the projection density across different subregions
                plt.figure(figsize=(6,2.4))
                vol_dict = {}
                for subregid in sub_ids:
                    vol = (self.cp_parc == subregid).sum() / 2 / 40**3
                    vol_dict[subregid] = vol
                # original scale of projection: total length
                #projs_o = np.exp(sub_projs) - 1
                projs_os = sub_projs.reset_index().melt(id_vars='index', var_name='Subregions', value_name='Projection')
                # re-ordering to match with clustermap
                sids = []
                for xtl in g.ax_heatmap.get_yticklabels():
                    sids.append(np.nonzero(projs_os.Subregions == xtl.get_text())[0])
                sids = np.hstack(sids)
                projs_os = projs_os.iloc[sids]
                g2 = sns.lineplot(projs_os, x='Subregions', y='Projection', 
                             markers=True, errorbar=('ci', 95), sort=False,
                             color='fuchsia')
                # to make it suitable for rotation
                plt.tick_params('x', labelrotation=90)
                plt.tick_params('y', left=False, labelleft=False, right=True, labelright=True, 
                                labelrotation=90)
                plt.xlim(-0.5, len(g.ax_heatmap.get_yticklabels())-0.5)
                plt.ylabel(r'$\ln(L+1)$')

                ax = plt.gca()
                ax.yaxis.set_label_position("right")
                plt.xlabel('')

                g2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
                plt.subplots_adjust(bottom=0.4, right=0.8)
                plt.savefig(f'proj_distribution_subregions_{tname}.png', dpi=300); plt.close()
                

       

if __name__ == '__main__':

    me_atlas_file = '../../intermediate_data/parc_r671_full_hemi2.nrrd'
    me_proj_csv = 'cp_1876_proj_ccf-me.csv'
    atlas_file = '../../intermediate_data/parc_r671_full.nrrd'
    proj_csv = 'cp_1876_proj.csv'
    axon_dir = './cp_axons'

    if 0:
        # generate the projection matrix
        axon_files = glob.glob(os.path.join(axon_dir, '*.swc'))
        PJ = Projection(resample_scale=1., atlas_file=me_atlas_file)
        PJ.calc_proj_matrix(axon_files, proj_csv=me_proj_csv)

    if 0:
        # estimate the subregional distribution of somas in CP
        cp_parc_file = '../../output_full_r671/parc_region672.nrrd'
        EP = EvalParcellation(cp_parc_file)
        EP.to_proj(proj_csv)

    if 1:
        file_me2ccf = f'{atlas_file}.pkl'
        meta_table = 'TableS6_Full_morphometry_1222.xlsx'
        EP = EvalParcellation(me_atlas_file)
        EP.projected_subregions(me_proj_csv, file_me2ccf, meta_table=meta_table)

