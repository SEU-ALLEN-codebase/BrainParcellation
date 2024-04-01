##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-25
#Description:               
##########################################################
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import random
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

from file_io import load_image
from anatomy.anatomy_config import SALIENT_REGIONS, REGION314, MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree


def cortical_separability(mefile, regions, sname, disp_type, cnt_thres=20):
    
    laminations = ['1', '2/3', '4', '5', '6a', '6b']
    
    # All sub-regions
    sregions = []
    for region in regions:
        for lam in laminations:
            sregions.append(region + lam)

    t0 = time.time()
    # load the microenvironment features and their affiliated meta informations
    df = pd.read_csv(mefile, comment='#', index_col=0)
    dfc = df[df.region_name_r671.isin(sregions)]

    rnames, rcnts = np.unique(dfc.region_name_r671, return_counts=True)
    # do not take into consideration of low-count regions
    rnames = rnames[rcnts > cnt_thres]
    # re-select the neurons
    dfc = dfc[dfc.region_name_r671.isin(rnames)]
    r671_names = dfc.region_name_r671
    lams = [r1[len(r2):] for r1, r2 in zip(r671_names, dfc.region_name_r316)]

    #feat_names = [fname for fname in dfc.columns if fname[-3:] == '_me']
    __FN__ = ('Length', 'AverageFragmentation', 'AverageContraction')
    feat_names = [fname+'_me' for fname in __FN__]
    fnames = feat_names + ['region_name_r316', 'region_name_r671']
    dfc = dfc[fnames]
    dfc['lamination'] = lams
    print(f'--> Time used in loading and processing data: {time.time() - t0:.2f} seconds')

    # visualize using pairplot
    if disp_type == 'l':
        hue = 'lamination'
        hue_order = laminations
        figname = f'lamination_{sname}.png'
    elif disp_type == 'r316':
        hue = 'region_name_r316'
        hue_order = regions
        figname = f's-regions_{sname}.png'
    elif disp_type == 'r671':
        hue = 'region_name_r671'
        hue_order = rnames
        figname = f'sl_regions_{sname}.png'

    g = sns.pairplot(dfc, vars=feat_names, hue=hue, plot_kws={'marker':'.'}, 
                     kind='scatter', diag_kws={'common_norm':False},
                     hue_order=hue_order)
    handles = g._legend_data.values()
    labels = g._legend_data.keys()
    g._legend.remove()
    g.fig.legend(handles=handles, labels=labels, markerscale=2, loc='center right')
    
    plt.savefig(figname, dpi=300)
    plt.close()
    

    #mat_mean = dfc[[fname+'_me' for fname in __FN__] + ['region_name_r671']].groupby('region_name_r671').mean()
    #mat_mean = StandardScaler().fit_transform(mat_mean)


def get_struct(id_path, bstructs):
    for idx in id_path[::-1]:
        if idx in bstructs:
            return idx
    else:
        return 0

def dsmatrix_of_all(mefile, ds_file1, ds_file2, metric='euclidean', cnt_thres=20):
    __FN__ = ('Length', 'AverageFragmentation', 'AverageContraction')
    feat_names = [fname+'_me' for fname in __FN__]
    random.seed(1024)

    # helper functions
    def dsmatrix(mefile, ds_file1, ds_file2, cnt_thres, feat_names, metric):
        df = pd.read_csv(mefile, comment='#', index_col=0)
        na_mask = df.region_name_r671.isna()
        # get the salient regions
        ana_tree = parse_ana_tree(keyname='id')
        salient_names = [ana_tree[idx]['acronym'] for idx in SALIENT_REGIONS]
        # in regions
        in_region = df.region_name_r671.isin(salient_names)
        fnames = feat_names + ['region_name_r671']
        # remove regions with insufficient neurons
        dfc = df[~na_mask & in_region][fnames]
        rs, rcs = np.unique(dfc.region_name_r671, return_counts=True)
        rcs_m = rcs >= cnt_thres
        rs = rs[rcs_m]
        rcs = rcs[rcs_m]
        dfc = dfc[dfc.region_name_r671.isin(rs)]
        # standardize
        tmp = dfc[feat_names].copy()
        tmp = (tmp - tmp.mean()) / tmp.std()
        if metric == 'cosine':
            tmp = tmp / (np.linalg.norm(tmp.values, axis=1, keepdims=True) + 1e-10)

        # estimate DS matrix
        t0 = time.time()
        dsm1 = np.zeros((len(rs), len(rs)))
        dsm2 = np.zeros((len(rs), len(rs)))
        for irs, rsi in enumerate(rs):
            rsm1 = tmp[dfc.region_name_r671 == rsi]
            for jrs in range(irs, len(rs)):
                rsj = rs[jrs]
                rsm2 = tmp[dfc.region_name_r671 == rsj]
                pdists = pairwise_distances(rsm1, rsm2, metric=metric)
                pm = pdists.mean()
                ps = pdists.std()
                dsm1[irs, jrs] = pm
                dsm1[jrs, irs] = pm
                dsm2[irs, jrs] = ps
                dsm2[jrs, irs] = ps
                
            print(f'[{irs}/{len(rs)}]: {time.time() - t0:.2f} seconds')

        d1 = pd.DataFrame(dsm1, columns=rs, index=rs)
        d2 = pd.DataFrame(dsm2, columns=rs, index=rs)
        # to file
        d1.to_csv(ds_file1)
        d2.to_csv(ds_file2)

        return d1, d2
    # --- End of helper functions --- #


    # --------- Inter-neuronal similarity ----------- #
    if os.path.exists(ds_file1) and os.path.exists(ds_file2):
        d1 = pd.read_csv(ds_file1, index_col=0)
        d2 = pd.read_csv(ds_file2, index_col=0)
    else:
        d1, d2 = dsmatrix(mefile, ds_file1, ds_file2, cnt_thres, feat_names, metric)
    print(d1.values.max(), d1.values.min())
       
    # Map the regions to brain structures
    '''
    bstructs = {
                688: 'CTX' 
                623: 'CNU',# STR + PAL
                512: 'CB',
                343: 'BS'   #: IB(549(TH)+HY), MB, HB
                }
    '''
    bstructs = {
        519: "CBN",
        528: "CBX", 
        703: "CTXsp", 
        1089: "HPF", 
        1097: "HY", 
        315: "Isocortex", 
        313: "MB", 
        354: "MY", 
        688: "OLF", 
        771: "P", 
        803: "PAL", 
        477: "STR", 
        549: "TH",
    }
    

    keepr = []
    ana_tree_n = parse_ana_tree(keyname='name')
    struct_ids = bstructs.keys()
    for region in d1.index:
        id_path = ana_tree_n[region]['structure_id_path']
        sid = get_struct(id_path, struct_ids)
        if sid == 0:
            keepr.append('')
        else:
            keepr.append(bstructs[sid])
    keepr = np.array(keepr)
    keepr_nz = (keepr!='').nonzero()[0]
    keepr = keepr[keepr_nz]
    d1 = d1.iloc[keepr_nz, keepr_nz]
    d2 = d2.iloc[keepr_nz, keepr_nz]

    keepr_s = pd.Series(keepr, name='structure')
    lut = dict(zip(np.unique(keepr_s), sns.hls_palette(len(np.unique(keepr_s)), l=0.5, s=0.8)))
    col_colors = keepr_s.map(lut)

    if metric == 'cosine':
        vmin, vmax = 0, 1
    elif metric == 'euclidean':
        vmin, vmax = 1, 4
    g = sns.clustermap(d1.reset_index(drop=True), cmap='hot_r', row_colors=col_colors, 
                       vmin=vmin, vmax=vmax, metric='euclidean') 
    for label in np.unique(keepr_s):
        g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=2)
    reordered_ind = g.dendrogram_col.reordered_ind
    
    plt.savefig('tmp.png')
    plt.close()

    #---- Section 2: interareal similarity
    d1c = d1.copy()
    d1c.columns = keepr_s
    d1c.index = keepr_s
    # in case no area found
    #import ipdb; ipdb.set_trace()
    bst = np.unique(d1c.index)
    interareal = np.zeros((len(bst), len(bst)))
    for si, s1 in enumerate(bst):
        for sj in range(si, len(bst)):
            s2 = bst[sj]
            vs = d1c.loc[s1, s2].values.mean()
            interareal[si, sj] = vs
            interareal[sj, si] = vs
    dfi = pd.DataFrame(interareal, index=bst, columns=bst)
    #sns.heatmap(dfi, cmap='afmhot')
    sns.clustermap(dfi)
    plt.savefig('tmp2.png')
    plt.close()
    

    # plot the correlation between distance and similarity
    df = pd.read_csv(mefile, comment='#', index_col=0)
    tmp = df[feat_names].copy()
    tmp = (tmp - tmp.mean()) / tmp.std()
    dfc = df.copy()
    dfc[feat_names] = tmp
    nsel = 5000
    sel_ids = random.sample(range(dfc.shape[0]), nsel)
    df_sel = dfc.iloc[sel_ids]
    pdists = pdist(df_sel[feat_names])
    cdists = pdist(df_sel[['soma_x', 'soma_y', 'soma_z']]/1000.)
    ## for debug
    cm = cdists < 1
    pdists = pdists[cm]
    cdists = cdists[cm]
    #
    nsel2 = 20000 
    sel_ids2 = random.sample(range(pdists.shape[0]), nsel2)
    cdists_sel = cdists[sel_ids2]
    pdists_sel = pdists[sel_ids2]
    sns.scatterplot(x=cdists_sel, y=pdists_sel, marker='.')
    stride = 0.05

    p75s = []
    for cs1 in np.arange(0, 1., stride):
        cs2 = cs1 + stride
        csm = (cdists_sel > cs1) & (cdists_sel <= cs2)
        pts = pdists_sel[csm]
        p75s.append(np.percentile(pts, 50))
    #import ipdb; ipdb.set_trace()
    plt.plot(np.arange(0, 1, stride)+stride//2, p75s, 'o-r')
    
    plt.savefig('tmp3.png')
    plt.close()


    # ------ Section 4: comparison between ds_inter and ds_intra ------- #
    ds_intra = np.diagonal(d1)
    ds_inter = d1.values[np.triu_indices_from(d1, k=1)]
    sns.kdeplot(ds_inter, label='inter-region', alpha=0.5, color='orange', fill=True)
    sns.kdeplot(ds_intra, label='intra-region', alpha=0.5, color='blue', fill=True)

    fig = plt.gcf()
    fig.set_size_inches(6,6)
    ax = plt.gca()
    __LABEL_FONTS__ = 18

    plt.xlabel(f'Distance in standardized feature space', fontsize=__LABEL_FONTS__)
    plt.ylabel(f'Density', fontsize=__LABEL_FONTS__)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2, direction='in', labelsize=__LABEL_FONTS__ - 4)
    ax.yaxis.set_tick_params(width=2, direction='in', labelsize=__LABEL_FONTS__ - 4)
    plt.legend(loc='upper right', frameon=False, fontsize=__LABEL_FONTS__)
    plt.savefig('tmp4.png', dpi=300)
    plt.close()


    
def analyze_dsmatrix_vs_parcellations(dsmean_file, parc_file, atlas_file=None):
    # load the ds matrix
    dsmean = pd.read_csv(dsmean_file, index_col=0)
    # load the standard CCFv3 atlas
    if atlas_file is None:
        atlas = load_image(MASK_CCF25_FILE)
    else:
        atlas = load_image(atlas_file)
    # load the annotation file
    ana_tree = parse_ana_tree(keyname='name')
    # load parcellation file
    parc = load_image(parc_file)

    # 
    ds_intra = np.diagonal(dsmean)
    ds_thr = 1.9
    high_ds = ds_intra > ds_thr
    high_ds_ids = high_ds.nonzero()[0]
    high_regs = dsmean.index[high_ds_ids]
    for ireg, reg in enumerate(high_regs):
        idx = ana_tree[reg]['id']
        atlas_mask = atlas == idx
        parc_mask = parc[atlas_mask]
        print(f'{reg}/ [DS: {ds_intra[high_ds_ids[ireg]]:.2f}]: {atlas_mask.sum()}, {np.unique(parc_mask)}')
    import ipdb; ipdb.set_trace()
    print()



if __name__ == '__main__':
    mefile = '../data/mefeatures_100K_with_PCAfeatures3.csv'

    if 0:
        region_dict = {
            'all':['FRP', 'MOp', 'MOs', 'SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr',
                   'SSp-un', 'SSs', 'GU', 'VISC', 'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal', 
                   'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'ACAd', 'ACAv',
                   'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl', 'AId', 'AIp', 'AIv', 'RSPagl', 'RSPd', 
                   'RSPv', 'VISa', 'VISrl', 'TEa', 'PERI', 'ECT'],
            'SSp':['SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr', 'SSp-un'],
            'VIS':['VISal', 'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'VISa', 'VISrl'],
            'AUD':['AUDd', 'AUDp', 'AUDpo', 'AUDv'],
            'AI':['AId', 'AIp', 'AIv']
        }
        
        for sname, regions in region_dict.items():
            for disp_type in ('r316', 'l'):
                cortical_separability(mefile, regions, sname, disp_type)

    if 0:
        #regions = ['FRP', 'MOp', 'SSp-n', 'SSs', 'GU', 'AUDd', 'VISp', 'ACAd', 'PL', 'ILA', 'ORBl', 'AId',
        #        'RSPd', 'TEa', 'PERI', 'ECT']
        regions = ['SSp-n']
        cortical_separability(mefile, regions, sname='r671', disp_type='r671')

    if 0:
        metric = 'euclidean'
        ds_file1 = f'dsmean_{metric}.csv'
        ds_file2 = f'dsstd_{metric}.csv'
        dsmatrix_of_all(mefile, ds_file1, ds_file2, metric=metric)

    if 1:
        dsmean_file = 'dsmean_euclidean.csv'
        parc_file = '../intermediate_data/parc_r671_full.nrrd'
        analyze_dsmatrix_vs_parcellations(dsmean_file, parc_file)
    

