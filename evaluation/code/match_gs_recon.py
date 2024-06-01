#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : match_gs_recon.py
#   Author       : Yufeng Liu
#   Date         : 2023-04-04
#   Description  : 
#
#================================================================

import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from swc_handler import get_soma_from_swc

gs_dir = '../data/1891_100um_2um_dendrite'
recon_dir = '/data/lyf/data/200k_v2/cropped_100um_resampled2um'
outfile = '../data/correspondence_gs_recons.csv'

# get all gold standard soma position
gs = []
cnt = 0
for swcfile in glob.glob(os.path.join(gs_dir, '*swc')):
    fn = os.path.split(swcfile)[-1]
    prefix = os.path.splitext(fn)[0]
    if fn.startswith('pre'):
        brain_id = fn.split('_')[1]
    else:
        brain_id = fn.split('_')[0]
    soma = get_soma_from_swc(swcfile, match_str='.*-1\n')
    spos = list(map(float, soma[2:5]))
    gs.append([prefix, brain_id, *spos, swcfile])
    
    cnt += 1
    if cnt % 100 == 0:
        print(cnt)
gs = pd.DataFrame(gs, columns=['filename', 'brain_id', 'xpos', 'ypos', 'zpos', 'path'])

# load all recon
res = []
scale = 1.

spos_file = '../data/recons_soma_pos.pkl'
spos_cached = os.path.exists(spos_file)
if spos_cached:
    with open(spos_file, 'rb') as fp:
        soma_dict = pickle.load(fp)
else:
    soma_dict = {}

rcnt = 0
for bdir in glob.glob(os.path.join(recon_dir, '[1-9]*')):
    brain_id = os.path.split(bdir)[-1]
    for swcfile in glob.glob(os.path.join(bdir, '*swc')):
        fn = os.path.split(swcfile)[-1]
        key = f'{brain_id}_{fn}'
        if spos_cached:
            xyz = soma_dict[key]
        else:
            soma = get_soma_from_swc(swcfile)
            xyz = np.array(list(map(float, soma[2:5]))) * scale
            soma_dict[key] = xyz
        res.append([fn, brain_id, *xyz, swcfile])

        rcnt += 1
        if rcnt % 100 == 0:
            print(rcnt)
    print(brain_id)
res = pd.DataFrame(res, columns=gs.columns)

if not spos_cached:
    with open(spos_file, 'wb') as fp:
        pickle.dump(soma_dict, fp)
 

dthr = 30
brains = np.unique(res.brain_id)
mindices = []
for brain in brains:
    print(brain)
    res_mask = res.brain_id == brain
    ridx = np.nonzero(res_mask.to_numpy())[0]
    res_i = res[res_mask]
    if brain in ['182712', '18467', '18469', '18866']:
        continue
    gs_mask = gs.brain_id == brain
    gidx = np.nonzero(gs_mask.to_numpy())[0]
    gs_i = gs[gs_mask]
    dm_i = distance_matrix(res_i[['xpos', 'ypos', 'zpos']], gs_i[['xpos', 'ypos', 'zpos']])
    print(dm_i.shape)
 
    if dm_i.shape[1] == 0: continue
    imin = dm_i.argmin(axis=1)
    vmin = dm_i.min(axis=1)
    # double check the size
    dflag = vmin < dthr
    fidx = np.nonzero(dflag)[0]
    
    pairs = [[ri, gi] for (ri, gi) in zip(ridx[dflag], gidx[imin][dflag])]
    mindices.extend(pairs)

mindices = np.array(mindices)
mres = res.iloc[mindices[:,0]].reset_index(drop=True)
mgs = gs.iloc[mindices[:,1]].reset_index(drop=True)
merged = pd.merge(mres, mgs, left_index=True, right_index=True)
merged.to_csv(outfile, index=False)


