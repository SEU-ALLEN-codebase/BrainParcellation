##########################################################
#Author:          Yufeng Liu
#Create time:     2024-06-06
#Description:               
##########################################################
import os
import glob
import random
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from swc_handler import parse_swc, write_swc, prune, find_soma_node
from morph_topo.morphology import Morphology

import sys
sys.path.append('../..')
from config import standardize_features, moranI_score

class MorphologyPerturbation:
    def __init__(self, swcfile, seed=1024):
        self.tree = parse_swc(swcfile)
        self.nsids = [i for i, node in enumerate(self.tree) if node[-6] != -1]
        self.sid = find_soma_node(self.tree)
        # deterministic
        random.seed(seed)
        np.random.seed(seed)

    def random_remove_points(self, npoints, outswcfile=None):
        sel_ids = random.sample(self.nsids, npoints)
        sel_indices = [self.tree[sel_id][0] for sel_id in sel_ids]
    
        # remove disconnected nodes
        pruned = prune(self.tree, set(sel_indices))
        print(len(self.tree), len(pruned))
        if outswcfile is not None:
            write_swc(pruned, outswcfile)

        return pruned

    def random_remove_tips(self, ratio, outswcfile=None):
        morph = Morphology(self.tree)
        try:
            _, seg_dict = morph.convert_to_topology_tree()
        except KeyError:
            if outswcfile is not None:
                write_swc(self.tree, outswcfile)
            return

        tips = list(morph.tips)

        # remove tips
        ndel = max(1, int(len(tips) * ratio))
        del_sid = random.sample(tips, ndel)
        
        del_nodes = []
        for sid in del_sid:
            del_nodes.append(sid)
            del_nodes.extend(seg_dict[sid])
        del_nodes = set(del_nodes)
    
        pruned = [node for node in self.tree if node[0] not in del_nodes]
        print(len(self.tree), len(pruned))
        if outswcfile is not None:
            write_swc(pruned, outswcfile)

        return pruned
        

def perturbate_folder(swcdir, outswcdir, ratio):
    if not os.path.exists(outswcdir):
        os.mkdir(outswcdir)
    
    t0 = time.time()
    for i, swcfile in enumerate(glob.glob(os.path.join(swcdir, '*swc'))):
        outswcfile = os.path.join(outswcdir, os.path.split(swcfile)[-1])
        mp = MorphologyPerturbation(swcfile)
        pruned = mp.random_remove_tips(ratio, outswcfile=outswcfile)
        
        if (i+1) % 20 == 0:
            print(time.time() - t0)

class FeatureEvolution:
    def __init__(self, lm_dir, feat_names=None, standardize=False,
                 rnames=['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']):
        if feat_names is None:
            import sys
            sys.path.append('../..')
            from config import mRMR_f3
            feat_names = mRMR_f3
        self.feat_names = feat_names
        self.coords_l, self.feats_l = self.load_data(lm_dir, rnames)

        sns.set_theme(style='ticks', font_scale=1.6)
        

    def load_data(self, lm_dir, rnames, standardize=False, remove_na=True):
        coords_l = []
        feats_l = []
        
        for ratio in np.arange(0, 0.9, 0.1):
            if ratio == 0:
                kstr = ''
            else:
                kstr = f'_del{ratio:.1f}'
            lm_file = os.path.join(lm_dir, f'lm_features_d28_dendrites{kstr}.csv')
            dfi = pd.read_csv(lm_file, index_col=0)
            dfi = dfi[dfi.region_name.isin(rnames)]
            # feature
            feats = dfi[self.feat_names]
            # standardization
            if standardize:
                standardize_features(feats, self.feat_names)
            feats_l.append(feats)

            # flipLR for coordinates
            coords = (dfi[['soma_x', 'soma_y', 'soma_z']] / 1000.).values
            zdim = 456
            zcoord = zdim * 25. / 1000
            right = np.nonzero(coords[:,2] > zcoord/2)[0]
            coords[right, 2] = zcoord - coords[right, 2]
            coords_l.append(coords)

            # remove the na data points
            if ratio == 0:
                na_flag = feats.isna().sum(axis=1) == 0
            else:
                na_flag = na_flag & (feats.isna().sum(axis=1) == 0)
        
        # filter all neurons
        print(na_flag.sum())
        for i in range(len(coords_l)):
            coords_l[i] = coords_l[i][na_flag]
            feats_l[i] = feats_l[i][na_flag]
        
        return coords_l, feats_l

    def plot_features(self):
        feats_l = []
        ratio_name = 'Deleted ratio'
        for i, feats in enumerate(self.feats_l):
            feats = feats.copy()
            feats[ratio_name] = i * 0.1
            feats_l.append(feats)
        feats_l = pd.concat(feats_l, ignore_index=True)
        feats_l.loc[:,'Length'] = feats_l['Length'] / 1000.
        # plot
        for fn in self.feat_names:
            sns.boxplot(data=feats_l, x=ratio_name, y=fn, 
                        width=0.35, color='black', fill=False)
            plt.xticks(ticks=np.arange(0,9,1), labels=[f'{r:.1f}' for r in np.arange(0, 0.9, 0.1)])
            if fn == 'Length':
                plt.ylabel('Length (mm)')
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(f'{fn}.png', dpi=300); plt.close()

    def plot_statistics(self):
        # Spatial auto-correlation
        moran_file = 'moran_cached.pkl'
        moran_cached = os.path.exists(moran_file)
        if moran_cached:
            with open(moran_file, 'rb') as fp:
                morans = pickle.load(fp)
        else:
            morans = []
            for coords, feats in zip(self.coords_l, self.feats_l):
                feats = feats.copy()
                standardize_features(feats, self.feat_names)
                moran = moranI_score(coords, feats.values, reduce_type='all')
                morans.append(moran)
                print(len(morans))
            # caching
            with open(moran_file, 'wb') as fp:
                pickle.dump(morans, fp)

        morans = np.array(morans)

        #----- STD -------#
        feats_all = pd.concat(self.feats_l)
        fmean, fstd = feats_all.mean(), feats_all.std()
        for feats in self.feats_l:
            feats = (feats - fmean) / fstd
            
        import ipdb; ipdb.set_trace()
        print()
        
        
    


if __name__ == '__main__':
    import time

    if 0:
        # random perturbation of swc files
        swcdir = './ION_HIP/swc_dendrites'
        ratio = 0.8
        outswcdir = f'./ION_HIP/swc_dendrites_del{ratio}'
        perturbate_folder(swcdir, outswcdir, ratio)

    if 1:
        lm_dir = 'ION_HIP'
        fe = FeatureEvolution(lm_dir)
        fe.plot_features()
        fe.plot_statistics()

