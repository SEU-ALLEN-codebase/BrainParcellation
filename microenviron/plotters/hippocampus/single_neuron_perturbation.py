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

    def random_remove_points(self, npoints, outswcfile=None, n_random=True):
        if npoints >= len(self.nsids):
            npoints = np.random.randint(0, len(self.nsids), 1)[0]
        elif n_random:
            npoints = np.random.randint(0, npoints, 1)[0]
        
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
        

def perturbate_folder(swcdir, outswcdir, ratio, seed=1024):

    if not os.path.exists(outswcdir):
        os.mkdir(outswcdir)
    
    t0 = time.time()
    for i, swcfile in enumerate(glob.glob(os.path.join(swcdir, '*swc'))):
        outswcfile = os.path.join(outswcdir, os.path.split(swcfile)[-1])
        mp = MorphologyPerturbation(swcfile, seed=seed)
        if type(ratio) is int:
            f_random = mp.random_remove_points
        else:
            f_random = mp.random_remove_tips
        pruned = f_random(ratio, outswcfile=outswcfile)
        
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
        
        ratios = [0,5,10,15,20,25,30,35,40]
        for ratio in ratios:
            if ratio == 0:
                kstr = ''
            else:
                kstr = f'_del_max{ratio}'
            lm_file = os.path.join(lm_dir, f'lm_features_d28_dendrites{kstr}.csv')
            if ratio == -1: # the original data
                lm_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
                dfi = pd.read_csv(lm_file, index_col=0)
                dfi = dfi[dfi.region_name_r671.isin(rnames)]
                __FEAT_NAMES = self.feat_names
            elif ratio == -2:
                lm_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
                dfi = pd.read_csv(lm_file, index_col=0)
                dfi = dfi[dfi.region_name_r671.isin(rnames)]
                __FEAT_NAMES = [f'{fn}_me' for fn in self.feat_names]
            else:
                dfi = pd.read_csv(lm_file, index_col=0)
                dfi = dfi[dfi.region_name.isin(rnames)]
                __FEAT_NAMES = self.feat_names
            # feature
            feats = dfi[__FEAT_NAMES]
            # standardization
            if standardize:
                standardize_features(feats, __FEAT_NAMES)
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
                na_ids = set(feats[feats.isna().sum(axis=1) == 0].index)
            elif ratio in [-1,-2]:
                continue
            else:
                na_ids = na_ids & set(feats[feats.isna().sum(axis=1) == 0].index)
        
        # filter all neurons
        for i, ratio in enumerate(ratios):
            if ratio in [-1,-2]:
                continue
            mask = feats_l[i].index.isin(na_ids)
            coords_l[i] = coords_l[i][mask]
            feats_l[i] = feats_l[i][mask]
        
        return coords_l, feats_l

    def plot_features(self):
        a2m_medians = {
            'Length': 0.933,
            'AverageContraction': 1.013,
            'AverageFragmentation': 0.891
        }

        feats_l = []
        ratio_name = 'Maximal deleted nodes'
        for i, feats in enumerate(self.feats_l):
            feats = feats.copy()
            feats[ratio_name] = i * 0.1
            feats_l.append(feats.values)
            print(i, feats.shape, feats.isna().sum().sum())
        feats_l = pd.DataFrame(np.vstack(feats_l), columns=self.feat_names+[ratio_name])
        feats_l.loc[:,'Length'] = feats_l['Length'] / 1000.
        # plot
        for fn in self.feat_names:
            sns.boxplot(data=feats_l, x=ratio_name, y=fn, 
                        width=0.35, color='black', fill=False)
            mval = feats_l[feats_l[ratio_name] == 0].median()[fn]
            hval = mval * a2m_medians[fn]
            print(mval, hval, a2m_medians[fn])
            plt.axhline(y=hval, linewidth=2, linestyle="--", color='r', clip_on=False)
            
            #plt.xticks(ticks=np.arange(0,9,1), labels=[f'{r:.1f}' for r in np.arange(0, 0.9, 0.1)])
            plt.xticks(ticks=np.arange(0,9,1), labels=[f'{r:d}' for r in range(0,40+1,5)])
            if fn == 'Length':
                plt.ylabel('Length (mm)')
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(f'{fn}.png', dpi=300); plt.close()

    def plot_statistics(self):
        # Spatial auto-correlation
        moran_files = ['moran_cached_point_perturbation_withME.pkl',
                       'moran_cached_point_perturbation_withME2.pkl',
                       'moran_cached_point_perturbation_withME3.pkl']
        #moran_files = ['moran_cached_point_perturbation_withME3.pkl']
        morans_all = []
        for moran_file in moran_files:
            moran_cached = os.path.exists(moran_file)
            if moran_cached:
                with open(moran_file, 'rb') as fp:
                    morans = pickle.load(fp)
            else:
                morans = []
                for coords, feats in zip(self.coords_l, self.feats_l):
                    feats = feats.copy()
                    fnames = feats.columns
                    standardize_features(feats, fnames)
                    moran = moranI_score(coords, feats.values, reduce_type='all')
                    morans.append(moran)
                    print(len(morans))
                # caching
                with open(moran_file, 'wb') as fp:
                    pickle.dump(morans, fp)

            morans_all.append(morans)

        morans_all = np.array(morans_all)
        print(morans_all.mean(axis=0))
        print(morans_all)

        length_morans = morans_all[:,:,0]
        xticks = range(0,40+1,5)
        
        plt.errorbar(xticks, length_morans.mean(axis=0), length_morans.std(axis=0), marker='o', markersize=8, linestyle='-', capsize=5, color='orchid', ecolor='black', linewidth=2)
        plt.xticks(xticks, labels=[f'{r:d}' for r in xticks])
        plt.xlabel('Maximal number of deleted nodes')
        plt.ylabel("Moran's Index")
        plt.subplots_adjust(bottom=0.15, left=0.20)
        plt.savefig('moran_perturbation.png', dpi=300); plt.close()

        #----- STD -------#
        feats_all = pd.concat(self.feats_l)
        fmean, fstd = feats_all.mean(), feats_all.std()
        for feats in self.feats_l:
            feats = (feats - fmean) / fstd
            
        #import ipdb; ipdb.set_trace()
        print()
        
        
    


if __name__ == '__main__':
    import time

    if 0:
        # random perturbation of swc files
        swcdir = './ION_HIP/swc_dendrites'
        for ratio in range(5,40+1,5):
            outswcdir = f'./ION_HIP/point_perturbation2/swc_dendrites_del_max{ratio}'
            perturbate_folder(swcdir, outswcdir, ratio, seed=1026)

    if 1:
        lm_dir = 'ION_HIP/point_perturbation3'
        fe = FeatureEvolution(lm_dir)
        #fe.plot_features()
        fe.plot_statistics()

