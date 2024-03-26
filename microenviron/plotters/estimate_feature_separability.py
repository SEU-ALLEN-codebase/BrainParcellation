##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-25
#Description:               
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import random

def cortical_separability(mefile, cnt_thres=20):
    
    regions = ['FRP', 'MOp', 'MOs', 'SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr',
               'SSp-un', 'SSs', 'GU', 'VISC', 'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal', 
               'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'ACAd', 'ACAv',
               'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl', 'AId', 'AIp', 'AIv', 'RSPagl', 'RSPd', 
               'RSPv', 'VISa', 'VISrl', 'TEa', 'PERI', 'ECT']
    
    #regions = ['SSp-n', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr', 'SSp-un']
    #regions = ['VISal', 'VISam', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor', 'VISa', 'VISrl']
    #regions = ['AUDd', 'AUDp', 'AUDpo', 'AUDv', 'VISal']
    regions = ['AId', 'AIp', 'AIv']
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
    hue = 'region_name_r316'
    #hue = 'lamination'
    g = sns.pairplot(dfc, vars=feat_names, hue=hue, plot_kws={'marker':'.'}, 
                     kind='scatter', diag_kws={'common_norm':False})
    plt.savefig('tmp.png', dpi=300)
    plt.close()
    

    #mat_mean = dfc[[fname+'_me' for fname in __FN__] + ['region_name_r671']].groupby('region_name_r671').mean()
    #mat_mean = StandardScaler().fit_transform(mat_mean)
    


if __name__ == '__main__':
    mefile = '../data/mefeatures_100K_with_PCAfeatures3.csv'

    if 1:
        cortical_separability(mefile)
    

