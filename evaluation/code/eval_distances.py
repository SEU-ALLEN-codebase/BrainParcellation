##########################################################
#Author:          Yufeng Liu
#Create time:     2024-06-01
#Description:               
##########################################################
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from swc_handler import parse_swc
from morph_topo import morphology

# Sholl analysis
def sholl_analysis(swc, max_radius=100, radius_incr=5):
    tree = parse_swc(swc)
    morph = morphology.Morphology(tree)
    spos = np.array(morph.pos_dict[morph.idx_soma][2:5])
    # all branching points
    bifs = morph.bifurcation
    bif_pos = np.array([morph.pos_dict[idx][2:5] for idx in bifs])
    # estimate the distances
    bif_rpos = bif_pos - spos.reshape((1,-1))
    bif_dists = np.linalg.norm(bif_rpos, axis=1)
    # histogram
    hcounts, hbins = np.histogram(bif_dists, np.arange(0, max_radius+radius_incr, radius_incr))
    
    hcenters = hbins[:-1] + radius_incr/2.
    return hcounts, hcenters

def compare_sholl(match_file, gs_dir, rec_dir, radius_incr=1):
    dfm = pd.read_csv(match_file, index_col=0, sep=' ')
    hcnts1 = []
    hcnts2 = []
    for irow, row in dfm.iterrows():
        rec = row.o_name
        gs_file = os.path.join(os.path.join(gs_dir, f'{irow}.swc'))
        rec_file = glob.glob(os.path.join(rec_dir, '*', f'{rec}_stps.swc'))[0]

        if (not os.path.exists(gs_file)) or (not os.path.exists(rec_file)):
            print(f'[Warning]: file {gs_file} or {rec_file} are not found!')
            continue
        
        hcnt1, hcen1 = sholl_analysis(gs_file, radius_incr=radius_incr)
        hcnt2, hcen2 = sholl_analysis(rec_file, radius_incr=radius_incr)
        hcnts1.append(hcnt1)
        hcnts2.append(hcnt2)

        if len(hcnts1) % 10 == 0:
            print(f'--> {len(hcnts1)}')
        #if len(hcnts1) == 50:
        #    break
    hcnts1 = np.array(hcnts1)
    hcnts2 = np.array(hcnts2)
    
    hist_total1 = hcnts1.sum(axis=0)
    hist_total1 = hist_total1 / hist_total1.sum()
    hist_total2 = hcnts2.sum(axis=0)
    hist_total2 = hist_total2 / hist_total2.sum()
    hist_bins = hcen1
    import ipdb; ipdb.set_trace()

    #df = pd.DataFrame(np.vstack((hist_bins, hist_total1, hist_total2)).transpose(), 
    #                  columns=('Distance-to-soma (µm)', 'Manual', 'Auto'))
    # visualize
    sns.set_theme(style='ticks', font_scale=1.6)
    #sns.lineplot(hist_bins, x='Distance-to-soma (µm)', y='')
    plt.plot(hist_bins, hist_total1, 'o-', color='blueviolet', markersize=3, label='Manual')
    plt.plot(hist_bins, hist_total2, 'o-', color='orange', markersize=3, label='Auto')
    plt.legend(frameon=False)
    plt.xlabel('Distance-to-soma (µm)')
    plt.ylabel('Proportion')
    plt.subplots_adjust(bottom=0.15, left=0.18)
    plt.savefig('sholl_branch.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    match_file = '../data/so_match_table.txt'
    rec_dir = '/data/lyf/data/200k_v2/cropped_100um_resampled2um/'
    gs_dir = '../../microenviron/data/S3_2um_dendrite/'
    compare_sholl(match_file, gs_dir, rec_dir)

