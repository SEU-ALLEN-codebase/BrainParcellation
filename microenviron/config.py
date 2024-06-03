##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
import numpy as np
import pysal.lib as pslib
from esda.moran import Moran

mRMR_f3 = ['Length', 'AverageContraction', 'AverageFragmentation']
mRMR_f3me = ['Length_me', 'AverageContraction_me', 'AverageFragmentation_me']

BS7_COLORS = {
    'CTX': 'limegreen',
    'CNU': 'darkorange',
    'CB': 'royalblue',
    'TH': 'violet',
    'MB': 'sienna',
    'HY': 'mediumslateblue',
    'HB': 'red'
}

def standardize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

def normalize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.min()) / (fvalues.max() - fvalues.min() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

def gini_coeff(points):
    points = np.array(points)
    n = len(points)
    diff_sum = np.sum(np.abs(points[:, None] - points))
    return diff_sum / (2 * n * np.sum(points))

def moranI_score(coords, feats, eval_ids=None, reduce_type='average'):
    # spatial coherence
    weights = pslib.weights.DistanceBand.from_array(coords, threshold=0.5)
    avgI = []
    if eval_ids is None:
        eval_ids = range(feats.shape[1])
    for i in eval_ids:
        moran = Moran(feats[:,i], weights)
        avgI.append(moran.I)
    
    if reduce_type == 'average':
        avgI = np.mean(avgI)
    else:
        avgI = np.max(avgI)
    return avgI

