##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-02
#Description:               
##########################################################
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from swc_handler import get_soma_from_swc
from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from projection.projection import Projection


sns.set_theme(style='ticks', font_scale=1.6)

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

        import ipdb; ipdb.set_trace()
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
        
        
        names = [f'R{i+1}' for i in range(mcorr.shape[0])]
        df_mcorr = pd.DataFrame(mcorr, index=names, columns=names)
        sns.clustermap(df_mcorr, cmap='hot_r')
        plt.savefig('subregion_vs_projection.png', dpi=300); plt.close()
        
        #import ipdb; ipdb.set_trace()
        print()
        

if __name__ == '__main__':

    proj_csv = 'cp_1876_proj.csv'
    axon_dir = './cp_axons'

    if 0:
        # generate the projection matrix
        axon_files = glob.glob(os.path.join(axon_dir, '*.swc'))
        PJ = Projection(resample_scale=1.)
        PJ.calc_proj_matrix(axon_files, proj_csv=proj_csv)

    if 1:
        cp_parc_file = '../../output_full_r671/parc_region672.nrrd'
        EP = EvalParcellation(cp_parc_file)
        EP.to_proj(proj_csv)

