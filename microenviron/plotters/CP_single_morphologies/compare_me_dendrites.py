##########################################################
#Author:          Yufeng Liu
#Create time:     2024-06-05
#Description:               
##########################################################
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
from config import mRMR_f3, mRMR_f3me, moranI_score, load_features, standardize_features

# compare the statistics of ME features and original dendritic features
feat_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
rname = 'CP'

df, fnames = load_features(feat_file, feat_type='full')
df_cp = df[df.region_name_r671 == rname]
fn22_me = [col for col in df.columns if col.endswith('_me')]
fn22 = [col[:-3] for col in fn22_me]

df_coords = df_cp[['soma_x', 'soma_y', 'soma_z']] / 40. # to mm
df_me = df_cp[fn22_me]; standardize_features(df_me, fn22_me)
df_de = df_cp[fn22]; standardize_features(df_de, fn22)

# estimate the spatial coherence
# the calculation of MoranI is very time-costly, use a subset
use_subset = True
nsel = 5000
if use_subset and (df_me.shape[0] > nsel):
    random.seed(1024)
    sel_ids = np.array(random.sample(range(df_me.shape[0]), nsel))
    coords = df_coords.iloc[sel_ids].values
    mes = df_me.iloc[sel_ids].values
    des = df_de.iloc[sel_ids].values
        
else:
    coords = df_coords.values
    mes = df_me.values
    des = df_de.values


moranI_me = np.array(moranI_score(coords, mes, reduce_type='all'))
moranI_de = np.array(moranI_score(coords, des, reduce_type='all'))
print(f'Avg Moran Index for ME and DE are {moranI_me.mean():.2f}, {moranI_de.mean():.2f}')

sns.set_theme(style='ticks', font_scale=1.6)
fig, ax = plt.subplots(figsize=(6,6))
plt.plot(moranI_me, range(len(moranI_me)), 'o-', color='orange', lw=2)
plt.plot(moranI_de, range(len(moranI_de)), 'o-', color='blueviolet', lw=2)
plt.xlabel("Moran's Index")
# processing the feature labels
pf2label = {
    'AverageBifurcationAngleRemote': 'Bif angle remote',
    'AverageBifurcationAngleLocal': 'Bif angle local',
    'AverageContraction': 'Contraction',
    'AverageFragmentation': 'Avg. Fragmentation',
    'AverageParent-daughterRatio': 'Avg. PD ratio',
    'Bifurcations': 'No. of bifs',
    'Branches': 'No. of branches',
    'HausdorffDimension': 'Hausdorff dimension',
    'MaxBranchOrder': 'Max. branch order',
    'Length': 'Total length',
    'MaxEuclideanDistance': 'Max. Euc distance',
    'MaxPathDistance': 'Max. path distance',
    'Nodes': 'No. of nodes',
    'OverallDepth': 'Overall z span',
    'OverallHeight': 'Overall y span',
    'OverallWidth': 'Overall x span',
    'Stems': 'No. of stems',
    'Tips': 'No. of tips',
}
fn_ticks = []
for name in fn22:
    if name in pf2label:
        fn_ticks.append(pf2label[name])
    else:
        fn_ticks.append(name)
plt.yticks(ticks=range(len(fn_ticks)), labels=fn_ticks, fontsize=12)

axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(2)
axes.spines['left'].set_linewidth(2)
axes.xaxis.set_tick_params(width=2, direction='in')
axes.yaxis.set_tick_params(width=2, direction='in')
#plt.setp(axes.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')


plt.subplots_adjust(left=0.38, bottom=0.12)
plt.ylabel('Morphological feature')

#plt.legend(frameon=False, loc='upper center')
plt.savefig(f'MoranI_improvement_of_{rname}.png', dpi=300); plt.close()

print()

