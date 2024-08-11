#!/usr/bin/env python
# coding: utf-8
import glob
import os
import sys
import json
import time
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from file_io import load_image
from anatomy.anatomy_config import REGION671, MASK_CCF25_FILE


data_path = '/home/sujun/parcellation_data'
t0 = time.time()
parc_id_table_path = os.path.join(data_path, 'parcellation_to_parcellation_term_membership_acronym.csv')
print(f'--> [{time.time() - t0:.2f} sec] Loading the parcellation hierarchy...')
parc_id_table = pd.read_csv(parc_id_table_path)

parc_id_match = {}
id_parc_match = {}
for i in range(len(parc_id_table)):
    parc_id_match[parc_id_table['parcellation_index'].iloc[i]] = parc_id_table['substructure'].iloc[i]
    id_parc_match[parc_id_table['substructure'].iloc[i]] = parc_id_table['parcellation_index'].iloc[i]

# processed transcript matrix for each cell (n=~1.1M)
print(f'--> [{time.time() - t0:.2f} sec] Parsing the gene expression data...')
with open(os.path.join(data_path, 'screened1_matrix.csv.pkl'), 'rb') as fp:
    T1 = pickle.load(fp)
# meta information for the cells (n=~2.8M)
print(f'--> [{time.time() - t0:.2f} sec] Parsing the meta information for each cell')
with open(os.path.join(data_path, 'cell_metadata_with_cluster_annotation_1.csv.pkl'), 'rb') as fp:
    c1_table = pickle.load(fp)
print(f'<-- [{time.time() - t0:.2f} sec] finished parsing the data')

# the annotation file
print(f'--> [{time.time() - t0:.2f} sec] Parsing the annotation file...')
annotation_table_path = os.path.join(data_path, '41586_2023_6812_MOESM8_ESM.xlsx.pkl')
with open(annotation_table_path, 'rb') as fp:
    annotation_table = pickle.load(fp)
print(f'<-- [{time.time() - t0:.2f} sec] Finish parsing the annotation file')


# subclass & neighborhood in Yao et al., 2023
subclass_to_neighborhood = {}
for i in np.unique(annotation_table['subclass_id']):
     neigh_rowid = annotation_table[annotation_table['subclass_id'] == i].index[0]
     neigh = annotation_table['neighborhood'].iloc[neigh_rowid]
     subclass_to_neighborhood[i] = neigh

parc_id_match_reverse = {}
for k,v in parc_id_match.items():
    parc_id_match_reverse[v] = k

tree_json_path = os.path.join(data_path, 'tree_yzx.json')
with open(tree_json_path, 'r') as file:
    tree_json = json.load(file) 

brain_to_id = {}
id_to_brain = {}
brain_id_path = {}
for i in range(len(tree_json)):
    bid = tree_json[i]['id']
    acronym = tree_json[i]['acronym']
    p = tree_json[i]['structure_id_path']
    if bid not in brain_to_id:
        brain_to_id[bid] = acronym
        id_to_brain[acronym] = bid
    if bid not in brain_id_path:
        brain_id_path[bid] = p 

major_BS = [688,549,623,313,1065,512,1097]  # Brain structures/areas
major_BS_name = ['CTX','TH','CNU','MB','HB','CB','HY','others']
region_Rough = {}
for i in REGION671:
    tmp_p = brain_id_path[i]
    flag = False
    for j in range(len(major_BS)):
        if major_BS[j] in tmp_p:
            region_Rough[brain_to_id[i]] = major_BS_name[j]
            flag = True
    if flag == False:
        region_Rough[brain_to_id[i]] = major_BS_name[len(major_BS_name)-1]

R_name_list = []
parc_path = '../../output_full_r671'
merged_parc_path = '../../intermediate_data/parc_r671_full.nrrd'
for f in glob.glob(parc_path+"/*"):
    f_name = os.path.basename(f)
    region_id = int(f_name.split(".")[0].split("parc_region")[1])
    R_name = brain_to_id[region_id]
    R_name_list.append(R_name)

expression_df = []
selected_R = []
exp_regions = []
exp_parc_ids = []
Gclass_label = []
Gsubclass_label = []
Gcluster_label = []
# transcripts
col_s = T1.columns[1:1123]

print(f'--> [{time.time() - t0:.2f} sec] Groupby the CCF-ME atlas')
full_parc = load_image(merged_parc_path)
ccf_atlas = load_image(MASK_CCF25_FILE)
ccf_atlas_salient = ccf_atlas.copy()
ccf_atlas_salient[full_parc < 0] = 0

t1 = time.time()
for n in R_name_list:
    print(n)
    if n not in parc_id_match_reverse:
        acronym_name = n+"-unassigned"
    else:
        acronym_name = n
    if acronym_name in parc_id_match_reverse:
        nid = parc_id_match_reverse[acronym_name]
        rows1 = T1[T1['parcellation_index']==nid]
        s_names = np.array(rows1['cell_label'])
        c_df = c1_table[c1_table['cell_label'].isin(s_names)]
        R_df_tmp = pd.merge(c_df,rows1,on='cell_label')
        
        if len(rows1) > 0:
            region_id = id_to_brain[n]
            mask_path = os.path.join(parc_path, 'parc_region'+str(region_id)+'.nrrd')
            if not os.path.exists(mask_path):
                print(region_id)
                continue
            
            mask_binary = ccf_atlas_salient == region_id
            parc_mask_tmp = np.zeros_like(full_parc)
            tmp2 = full_parc[mask_binary]
            parc_mask_tmp[mask_binary] = tmp2 - tmp2.min() + 1
            
            #parc_mask_tmp = load_image(mask_path)
            print(f'>> region={region_id}: {time.time() - t1:.2f} sec')

            tmp_df = R_df_tmp.copy()
            tmp_df['x_y'] = round(tmp_df['x_y']*40).astype(int)
            tmp_df['y_y'] = round(tmp_df['y_y']*40).astype(int)
            tmp_df['z_y'] = round(tmp_df['z_y']*40)
            tmp_df['z_y'] = tmp_df['z_y'].apply(lambda x: 456-x if x<228 else x)
            tmp_df['z_y'] = tmp_df['z_y'].astype(int)
            label_tmp = parc_mask_tmp[tmp_df['z_y'],tmp_df['y_y'],tmp_df['x_y']]
            # print(np.unique(label_tmp))
            gclass = tmp_df['class'].apply(lambda x:x.split(' ')[0])
            gsubclass = tmp_df['subclass'].apply(lambda x:x.split(' ')[0])
            gcluster = tmp_df['cluster_alias']
            for j in range(1,np.max(label_tmp)+1):
                rows_parc = tmp_df[label_tmp == j]
                if len(rows_parc) > 0:
                    exp_regions.append(n)
                    exp_parc_ids.append(j)
                    expression_df.append(np.array(rows_parc[col_s].mean()))
                    Gclass_label.append(gclass[label_tmp == j])
                    Gsubclass_label.append(gsubclass[label_tmp == j]) 
                    Gcluster_label.append(gcluster[label_tmp == j])   
            selected_R.append(n) 
            print(f'<< {time.time() - t1:.2f} sec')
print('<-- Finished grouping the expression data')

expression_df = np.array(expression_df)
S_exp1 = (expression_df-np.mean(expression_df,axis=0))/np.std(expression_df,axis=0)
scaler = preprocessing.Normalizer(norm='l2')
S_exp2 = scaler.fit_transform(S_exp1)

D_exp = cdist(S_exp2,S_exp2,metric = 'cosine')
D_exp2 = (D_exp.max()-D_exp)/D_exp.max()

R_gene_parc = [region_Rough[x] for x in exp_regions]


cmap = mpl.cm.rainbow
color_hm = cmap(np.linspace(0,1,len(major_BS_name)))
color_hm2 = [mpl.colors.rgb2hex(i,keep_alpha=False) for i in color_hm]
c = dict(zip(major_BS_name,color_hm2))
row_c = list(map(lambda x:c[x], R_gene_parc))
# sns.set(style="ticks", rc={'ytick.right':False,'ytick.left':False,'xtick.bottom':False,'xtick.top':False})
g_exp = sns.clustermap(data = D_exp2,
                       row_colors = row_c,
                       cmap='BuGn'
                       )
g_exp.ax_heatmap.set_xticks(np.arange(len(exp_regions)))
g_exp.ax_heatmap.set_yticks(np.arange(len(exp_regions)))

xticklabels = g_exp.data2d.columns
yticklabels = g_exp.data2d.columns

interval = 10
g_exp.ax_heatmap.set_xticklabels(
     [exp_regions[int(label)] if i % interval == 0 else '' for i,label in enumerate(xticklabels)],
     rotation=70,fontsize=5
)
g_exp.ax_heatmap.set_yticklabels(
     [exp_regions[int(label)] if i % interval == 0 else '' for i,label in enumerate(yticklabels)],
     rotation=0,fontsize=5
)

g_exp.ax_heatmap.tick_params(length=0)
plt.savefig('figures/gene_matrix_full2_avgLog.png',dpi=300)

class_stats = np.zeros((len(D_exp2),29))
reordered_index = g_exp.dendrogram_col.reordered_ind
for i in range(len(reordered_index)):
     class_rows = np.array(Gclass_label[reordered_index[i]])
     tmp_classes = np.unique(class_rows)
     total_N = len(class_rows)
     for j in tmp_classes:
          n = len(class_rows[class_rows == j])
          class_stats[i,int(j)-1] = n/total_N*100

class_stats = np.array(class_stats)

cg = sns.clustermap(class_stats,row_cluster=False,col_cluster=True)

reordered_cg = cg.dendrogram_col.reordered_ind

class_stats_reorder = class_stats[:,reordered_cg]

plt.figure(figsize=(3,8))
sns.heatmap(class_stats_reorder,cmap='Reds')
plt.savefig('figures/Gclass.png',dpi=300)


ng = sns.clustermap(neighborhood_stats,row_cluster=False,col_cluster=True)

ng_reorder = ng.dendrogram_col.reordered_ind
neighborhood_stats_reorder = neighborhood_stats[:,ng_reorder]

plt.figure(figsize=(1,8))
sns.heatmap(neighborhood_stats_reorder,cmap='Reds')
plt.savefig('figures/Gneighborhood.png',dpi=300)

sys.exit()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from esda.moran import Moran
import pysal.lib as pslib
# import warnings
from scipy.spatial import distance_matrix
def moran_calc2(num_cluster,labels, coords):
    # warnings.filterwarnings('ignore',category=UserWarning)
    N = len(coords)
    if N > 60000:    
        ids = [10*i for i in range(int(np.floor(N/10)))]
        coords = coords[ids,:]
        labels = np.array(labels)[ids]
    dist_m = distance_matrix(coords,coords)
    thres = np.percentile(dist_m,25)
    weights = pslib.weights.DistanceBand.from_array(coords, threshold=thres)
    moran = Moran(labels, weights)
    avgI = moran.I
    pI = moran.p_norm
    zI = moran.z_norm
    return avgI,pI,zI

T_info1 = pd.read_csv('gene/screened_info1.csv')

get_ipython().run_cell_magic('capture', '', '\nmoran_func = True\nif moran_func:\n    moran_r, moran_scores,moran_r_parc, moran_scores_parc, moran_parc_ids = [],[],[],[],[]\nelse:\n    std_r, std_scores,std_r_parc, std_scores_parc, std_parc_ids, std_delta_parc, cv_parc= [],[],[],[],[],[],[]\n\nfor i in REGION671:\n    mask_path = \'lyf/output_full_r671/parc_region\'+str(i)+\'.nrrd\'\n    if not os.path.exists(mask_path):\n        continue\n    parc_mask_tmp = load_image(mask_path)\n    N_parc = np.max(parc_mask_tmp)\n    if N_parc <2:\n        continue\n    region_id = i\n    acronym_name = brain_to_id[region_id]\n    if acronym_name not in id_parc_match:\n        acronym_name = acronym_name+"-unassigned"\n    if acronym_name not in id_parc_match:\n        continue\n    r_name = id_parc_match[acronym_name]\n    \n    if moran_func:\n        t_tmp_df = T_info1[T_info1[\'parcellation_index\']==r_name]\n        if len(t_tmp_df) < 10:\n            continue\n        s_names = np.array(t_tmp_df[\'cell_label\'])\n        c_tmp_df = c1_table[c1_table[\'cell_label\'].isin(s_names)]\n        R_df_tmp = pd.merge(c_tmp_df,t_tmp_df,on=\'cell_label\')\n        coords = R_df_tmp[[\'x_y\',\'y_y\',\'z_y\']].values\n        R_df_tmp[\'x_y\'] = round(R_df_tmp[\'x_y\']*40).astype(int)\n        R_df_tmp[\'y_y\'] = round(R_df_tmp[\'y_y\']*40).astype(int)\n        R_df_tmp[\'z_y\'] = round(R_df_tmp[\'z_y\']*40)\n        R_df_tmp[\'z_y\'] = R_df_tmp[\'z_y\'].apply(lambda x: 456-x if x<228 else x)\n        R_df_tmp[\'z_y\'] = R_df_tmp[\'z_y\'].astype(int)    \n        g_labels = R_df_tmp[\'cluster\'].apply(lambda x: int(x.split(\' \')[0]))\n        num_cluster = len(np.unique(g_labels))\n        \n        parc_labels = parc_mask_tmp[R_df_tmp[\'z_y\'],R_df_tmp[\'y_y\'],R_df_tmp[\'x_y\']]\n    \n        moran_score,moran_pi,moran_zi = moran_calc2(num_cluster,g_labels, coords)\n        moran_scores.append(moran_score)\n        moran_r.append(brain_to_id[region_id])  \n        \n        del t_tmp_df,c_tmp_df, parc_mask_tmp\n    else:\n        t_tmp_df = T1[T1[\'parcellation_index\']==r_name]\n        if len(t_tmp_df) < 10:\n            continue\n\n        R_df_tmp = t_tmp_df\n\n        R_df_tmp[\'x\'] = round(R_df_tmp[\'x\']*40).astype(int)\n        R_df_tmp[\'y\'] = round(R_df_tmp[\'y\']*40).astype(int)\n        R_df_tmp[\'z\'] = round(R_df_tmp[\'z\']*40)\n        R_df_tmp[\'z\'] = R_df_tmp[\'z\'].apply(lambda x: 456-x if x<228 else x)\n        R_df_tmp[\'z\'] = R_df_tmp[\'z\'].astype(int)\n\n        parc_labels = parc_mask_tmp[R_df_tmp[\'z\'],R_df_tmp[\'y\'],R_df_tmp[\'x\']]\n     \n        ss = StandardScaler()\n        R_df_scaled = ss.fit_transform(R_df_tmp[col_s])\n        pca = PCA(n_components=3)\n        R_df_trans = pca.fit_transform(R_df_scaled)\n        std_r.append(brain_to_id[region_id])\n        top_std = np.std(R_df_trans,axis=0)\n        std_scores.append(np.mean(top_std))\n        \n        # col_sum = np.sum(R_df_tmp[col_s])\n        # col_sum_sorted = np.sort(col_sum)\n        # nc = len(col_sum_sorted)\n        # count_add = 0\n        # col_ids = []\n        # for c in range(3):\n        #     v = col_sum_sorted[nc-c-1]\n        #     idlist = np.where(col_sum == v)[0]\n        #     for d in idlist:\n        #         col_ids.append(col_s[d])\n        #         count_add += 1\n        #         if count_add  > 3:\n        #             break\n        #     if count_add > 3: \n        #         break\n        \n        # cv_overall = np.std(R_df_tmp[col_ids],axis=0)/np.mean(R_df_tmp[col_ids],axis=0)\n    \n        del t_tmp_df,parc_mask_tmp\n    \n    for j in range(1,N_parc+1):\n        if moran_func:\n            g2 = g_labels[parc_labels == j]\n            if len(g2) <10:\n                continue\n            num_cluster_tmp = len(np.unique(g2))\n            coords_tmp = coords[parc_labels == j,:]\n            moran_score2,moran_pi2,moran_zi2 = moran_calc2(num_cluster_tmp,g2, coords_tmp)\n            moran_scores_parc.append(moran_score2)\n            moran_parc_ids.append(j)\n            moran_r_parc.append(brain_to_id[region_id])\n            \n            del coords_tmp\n        else: \n            tmp_rows = R_df_trans[parc_labels == j]\n            if len(tmp_rows) <10:\n                continue\n            std_r_parc.append(brain_to_id[region_id])\n            std_parc_ids.append(j)\n        \n            top_std_tmp = np.std(tmp_rows,axis=0)\n            std_scores_parc.append(np.mean(top_std_tmp))\n            std_delta_parc.append(np.mean(top_std_tmp)-np.mean(top_std))\n            \n            tmp_rows2 = R_df_tmp[parc_labels == j][col_ids]\n            # cv_org = np.std(tmp_rows2,axis=0)/np.mean(tmp_rows2,axis=0)\n            # cv_parc.append(np.mean(cv_org-cv_overall))\n            del tmp_rows,tmp_rows2\n    del R_df_tmp\n')

def load_features(mefile, scale=25., feat_type='mRMR', flipLR=True):
    df = pd.read_csv(mefile, index_col=0, comment='#')
    
    if feat_type == 'full':
        cols = df.columns
        fnames = [fname for fname in cols if fname[-3:] == '_me']
    elif feat_type == 'mRMR':
        # Features selected by mRMR
        fnames = mRMR_f3me
    elif feat_type == 'PCA':
        fnames = ['pca_feat1', 'pca_feat2', 'pca_feat3']
    else:
        raise ValueError("Unsupported feature types")

    # standardize
    tmp = df[fnames]
    tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
    df[fnames] = tmp

    # scaling the coordinates to CCFv3-25um space
    df['soma_x'] /= scale
    df['soma_y'] /= scale
    df['soma_z'] /= scale
    # we should remove the out-of-region coordinates
    zdim,ydim,xdim = (456,320,528)   # dimensions for CCFv3-25um atlas
    in_region = (df['soma_x'] >= 0) & (df['soma_x'] < xdim) & \
                (df['soma_y'] >= 0) & (df['soma_y'] < ydim) & \
                (df['soma_z'] >= 0) & (df['soma_z'] < zdim)
    df = df[in_region]
    print(f'Filtered out {in_region.shape[0] - df.shape[0]}')

    if flipLR:
        # mirror right hemispheric points to left hemisphere
        zdim2 = zdim // 2
        nzi = (df['soma_z'] < zdim2).values.nonzero()
        loci = df.index[nzi]
        df.loc[loci, 'soma_z'] = zdim - df.loc[loci, 'soma_z']

    return df, fnames

me_file_path = 'BrainParcellation/microenviron/data/mefeatures_100K_with_PCAfeatures3.csv'
me_df_new, fnames_new = load_features(me_file_path, scale=25., feat_type='full', flipLR=True)

import pymrmr

### L2/3
def moran_calc_me(df, fnames):
    coords = df[['soma_x', 'soma_y', 'soma_z']]/40
    dist_m = distance_matrix(coords,coords)
    thres = np.percentile(dist_m,25)
    weights = pslib.weights.DistanceBand.from_array(coords.values, threshold=thres,binary=True)
    avgI = []
    pI = []
    zI = []
    for fn in fnames:
        moran = Moran(df[fn], weights)
        avgI.append(moran.I)
        pI.append(moran.p_norm)
        zI.append(moran.z_norm)
    return avgI,pI,zI

get_ipython().run_cell_magic('capture', '', "moran_func2 = True\nif moran_func2:\n    moran_regions,moran_parcID,moran_I_all,moran_me_I,moran_me_r= [],[],[],[],[]\nelse:\n    fstd_regions,fstd_parcID,fstd_I_all,fstd_me_I,fstd_me_r,fstd_delta_all,cv_parc_me= [],[],[],[],[],[],[]\n\nfor i in REGION671:\n    mask_file = 'lyf/output_full_r671/parc_region'+str(i)+'.nrrd'\n    if os.path.exists(mask_file):\n        mask = load_image(mask_file)\n        N_parc = np.max(mask)\n        \n        me_df_r = me_df_new[me_df_new['region_id_r671']==i]\n        print(len(me_df_r))\n        if len(me_df_r) < 10:\n            continue\n        if N_parc <2:\n            continue\n        \n        me_df_r2 = me_df_r.copy()\n        me_df_r2['soma_z'] = round(me_df_r2['soma_z']).astype(int)\n        me_df_r2['soma_x'] = round(me_df_r2['soma_x']).astype(int)\n        me_df_r2['soma_y'] = round(me_df_r2['soma_y']).astype(int)\n    \n        parc_labels = mask[me_df_r2['soma_z'],me_df_r2['soma_y'],me_df_r2['soma_x']]\n        \n        del mask\n        \n        if moran_func2:\n            me_df_r2 = me_df_r2[fnames_new]\n            me_df_r2.insert(loc=0,column='parc_id',value=parc_labels)\n        \n            mrmr_F = pymrmr.mRMR(me_df_r2,'MIQ',3)\n        \n            avgI_me,pI_me,zI_me = moran_calc_me(me_df_r, mrmr_F)\n            moran_me_I.append(np.mean(avgI_me))\n            moran_me_r.append(brain_to_id[i]) \n                \n        else:\n            tmp_scaled = StandardScaler().fit_transform(me_df_r[fnames_new])\n        \n            pca = PCA(n_components=3)\n            tmp_trans = pca.fit_transform(tmp_scaled)\n            top_std = np.std(tmp_trans,axis=0)\n            meanSTD = np.mean(top_std)\n            fstd_me_I.append(meanSTD)\n            fstd_me_r.append(brain_to_id[i]) \n            \n            me_df_r2 = me_df_r2[fnames_new]\n            me_df_r2.insert(loc=0,column='parc_id',value=parc_labels)\n        \n            mrmr_F = pymrmr.mRMR(me_df_r2,'MIQ',3)\n            \n            cv_overall = np.std(me_df_r[mrmr_F],axis=0)/np.mean(me_df_r[mrmr_F],axis=0)\n            del tmp_scaled\n\n        for j in range(1,N_parc+1):\n            if len(me_df_r[parc_labels == j]) < 10:\n                continue\n            \n            if moran_func2:\n                me_tmp = me_df_r[parc_labels == j]            \n                avgI,pI,zI = moran_calc_me(me_tmp, mrmr_F)\n                \n                moran_I_all.append(np.mean(avgI))\n                moran_regions.append(brain_to_id[i])\n                moran_parcID.append(j)\n                \n                del me_tmp\n            else:\n                me_tmp = tmp_trans[parc_labels==j,:]\n                top_std_tmp = np.std(me_tmp,axis=0)\n                fstd_I_all.append(np.mean(top_std_tmp))\n                fstd_regions.append(brain_to_id[i])\n                fstd_parcID.append(j)\n                fstd_delta_all.append(np.mean(top_std_tmp) - meanSTD)\n                \n                me_tmp2 = me_df_r[parc_labels == j][mrmr_F]\n                cv_org = np.std(me_tmp2,axis=0)/np.mean(me_tmp2,axis=0)\n                cv_parc_me.append(np.mean(cv_org-cv_overall))\n                del me_tmp\n        del me_df_r, me_df_r2\n")

### match rows
moran_I_all2 = []
moran_R_full2 = []
moran_me_I2 = []
moran_me_R2 = []
keep_r_ids = []
n = 0
keep_parc_ids = []
for i in moran_r:
    if i in moran_me_r:
        moran_me_I2.append(moran_me_I[moran_me_r.index(i)])
        moran_me_R2.append(region_Rough[i])
        ids_tmp = np.where(np.array(moran_regions)==i)[0]
        ids_tmp2 = np.where(np.array(moran_r_parc)==i)[0]
        p_tmp = np.array(moran_parcID)[ids_tmp]
        p_tmp2 = np.array(moran_parc_ids)[ids_tmp2]
        ids_set = set(p_tmp).intersection(set(p_tmp2))
        tmp_p = []
        for k in list(ids_set):
            tmp_p.append(ids_tmp[np.where(p_tmp == k)[0][0]])
        for j in tmp_p:
            moran_I_all2.append(moran_I_all[j])
            moran_R_full2.append(region_Rough[i])
            keep_parc_ids.append(ids_tmp2[np.where(p_tmp2 == moran_parcID[j])[0][0]])
        keep_r_ids.append(n)
    n += 1

### match rows
fstd_I_all2 = []
fstd_R_full2 = []
fstd_me_I2 = []
fstd_me_R2 = []
keep_r_ids2 = []
fstd_delta_all2 = []
cv_parc_me2 = []
n = 0
keep_parc_ids2 = []
for i in std_r:
    if i in fstd_me_r:
        fstd_me_I2.append(fstd_me_I[fstd_me_r.index(i)])
        fstd_me_R2.append(region_Rough[i])
        ids_tmp = np.where(np.array(fstd_regions)==i)[0]
        ids_tmp2 = np.where(np.array(std_r_parc)==i)[0]
        p_tmp = np.array(fstd_parcID)[ids_tmp]
        p_tmp2 = np.array(std_parc_ids)[ids_tmp2]
        ids_set = set(p_tmp).intersection(set(p_tmp2))
        tmp_p = []
        for k in list(ids_set):
            tmp_p.append(ids_tmp[np.where(p_tmp == k)[0][0]])
        for j in tmp_p:
            fstd_I_all2.append(fstd_I_all[j])
            fstd_R_full2.append(region_Rough[i])
            fstd_delta_all2.append(fstd_delta_all[j])
            cv_parc_me2.append(cv_parc_me[j])
            keep_parc_ids2.append(ids_tmp2[np.where(p_tmp2 == fstd_parcID[j])[0][0]])
        keep_r_ids2.append(n)
    n += 1

moran_r2 = np.array(moran_r)[keep_r_ids]
moran_scores2 = np.array(moran_scores)[keep_r_ids]
moran_r_parc2 = np.array(moran_r_parc)[keep_parc_ids]
moran_scores_parc2 = np.array(moran_scores_parc)[keep_parc_ids]

moran_R = []
for i in moran_r2:
    moran_R.append(region_Rough[i])
moran_R = np.array(moran_R)
moran_R2 = []
for i in moran_r_parc2:
    moran_R2.append(region_Rough[i])
moran_R2 = np.array(moran_R2)

moran_R_full2 = np.array(moran_R_full2)
moran_me_R2 = np.array(moran_me_R2)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

cmap = mpl.cm.rainbow
color_hm = cmap(np.linspace(0,1,len(major_BS_name)-1))
color_hm2 = [mpl.colors.rgb2hex(i,keep_alpha=False) for i in color_hm]

plt.figure(figsize=(10,2))
current = 0.5
for i in range(len(major_BS_name)-1):
    score_tmp = np.array(moran_me_I2)[moran_me_R2==major_BS_name[i]]
    score_tmp = score_tmp*100
    bin_positions = np.arange(current,len(score_tmp)+current)
    plt.bar(bin_positions,score_tmp,width =0.5,color = color_hm2[i],align='center',edgecolor=color_hm2[i])
    current += len(score_tmp)
# plt.xlabel('Moran I',fontsize=15)
# plt.ylabel('#Sub-parcellation',fontsize=15)
plt.axhline(10,color='magenta',linestyle='--',linewidth=1.2)
plt.axhline(30,color='dimgray',linestyle='--',linewidth=1.2)
plt.axhline(0,color='black',linewidth=1)
plt.tick_params(labelbottom=False,labelleft=True)
plt.yticks([25*i for i in range(-1,4)],[str(0.25*i) for i in range(-1,4)],fontsize = 18)
ax = plt.gca()
for spine in ['right','top','bottom']:
    ax.spines[spine].set_visible(False)
# plt.savefig('all_figs/Fig5/moran_me_regions.png',dpi=300)

plt.figure(figsize=(10,2))
current = 0.5
for i in range(len(major_BS_name)-1):
    score_tmp = np.array(moran_scores2)[moran_R==major_BS_name[i]]
    score_tmp = score_tmp*100
    # score_tmp = list(score_tmp)
    bin_positions = np.arange(current,len(score_tmp)+current)
    plt.bar(bin_positions,score_tmp,width =0.5,color = color_hm2[i],align='center',edgecolor=color_hm2[i])
    current += len(score_tmp)
# plt.xlabel('Moran I',fontsize=15)
# plt.ylabel('#Sub-parcellation',fontsize=15)
plt.axhline(10,color='magenta',linestyle='--',linewidth=1.2)
plt.axhline(30,color='dimgray',linestyle='--',linewidth=1.2)
plt.axhline(0,color='black',linewidth=1)
plt.tick_params(labelbottom=False,labelleft=True)
plt.yticks([25*i for i in range(-1,4)],[str(0.25*i) for i in range(-1,4)],fontsize = 18)
ax = plt.gca()
for spine in ['right','top','bottom']:
    ax.spines[spine].set_visible(False)
# plt.savefig('all_figs/Fig5/moran_regions.png',dpi=300)

