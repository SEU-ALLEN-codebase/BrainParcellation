##########################################################
#Author:          Yufeng Liu
#Create time:     2025-01-14
#Description:               
##########################################################
import os
import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('..')
from config import get_me_ccf_mapper

RMAP = {
    943: 'MOp23',
    362: 'MD',
    961: 'PIR'
}

target_regid = 961 #PMR(1025), ICc(811), MOp2/3(943), PIR(961), SIM(1007), MD(362), AId6a(783)
min_neurons = 8
me_file = '../plotters/whole-brain_projection/data/proj_ccf-me.csv'
me2ccf_file = '../intermediate_data/parc_r671_full.nrrd.pkl'
meta_file = '../plotters/whole-brain_projection/data/meta_all.csv'
BSTRUCTS = ('HY', 'Isocortex', 'STR', 'TH', 'HPF')

me_projs = pd.read_csv(me_file, index_col=0)
me_projs.columns = me_projs.columns.astype(int)
# load the atlas mapper
me2ccf, ccf2me = get_me_ccf_mapper(me2ccf_file)

# get the axons projected to target region
proj_me_regs = ccf2me[target_regid] + [-ii for ii in ccf2me[target_regid]]
tar_projs = me_projs[proj_me_regs]
tar_projs = tar_projs[tar_projs.sum(axis=1) > 0]

# get the region information
meta = pd.read_csv(meta_file, index_col=0)
orig_rnames = meta.loc[tar_projs.index].region_name_ccf
mprojs = tar_projs.copy()
mprojs['rname'] = meta.loc[tar_projs.index].region_name_ccf#.struct13_name
structs =  meta.loc[tar_projs.index].struct13_name

# Remove regions with few neurons
rnames, rcnts = np.unique(mprojs['rname'][~mprojs['rname'].isna()], return_counts=True)
rnames_p = rnames[rcnts > min_neurons]
rcnts_p = rcnts[rcnts > min_neurons]

print(f'#Neuron distributions across structures: {rnames_p}, {rcnts_p}')
sprojs = mprojs[mprojs.rname.isin(rnames_p)].groupby('rname').mean()
# get the brain structure
rsmap = {}
for n1, n2 in zip(orig_rnames, structs):
    if (n1 is not np.nan) and (n1 not in rsmap):
        rsmap[n1] = n2

# to log-space
sprojs = np.log(sprojs + 1)

# plot
sns.set_theme(style='ticks', font_scale=1.4)
fig = plt.figure(figsize=(6,6))
row_structs = [rsmap[rn] for rn in sprojs.index]
print(f'Brain structures: {np.unique(row_structs)}')
# 
lut_row = {bs:plt.cm.rainbow(each, bytes=False) 
           for bs, each in zip(BSTRUCTS, np.linspace(0, 1, len(BSTRUCTS)))}
row_colors = np.array([lut_row[bs] for bs in row_structs])
g1 = sns.clustermap(sprojs, cmap='coolwarm', yticklabels=1, 
                    col_cluster=False, row_colors=row_colors,
                    #cbar_kws={'boundaries':[0,2,4,6,8,10]}
                    )
# 
plt.savefig(f'proj_clustermap_to_{RMAP[target_regid]}.png', dpi=300); plt.close()

print()

