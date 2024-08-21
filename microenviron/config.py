##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
import numpy as np
import pandas as pd
import pickle
import pysal.lib as pslib
from esda.moran import Moran


mRMR_f3 = ['Length', 'AverageContraction', 'AverageFragmentation']
mRMR_f3me = ['Length_me', 'AverageContraction_me', 'AverageFragmentation_me']
__FEAT24D__ = [
    'Stems', 'Bifurcations', 'Branches', 'Tips', 'OverallWidth', 'OverallHeight',
    'OverallDepth', 'Length', 'Volume', 'MaxEuclideanDistance', 'MaxPathDistance',
    'MaxBranchOrder', 'AverageContraction', 'AverageFragmentation',
    'AverageParent-daughterRatio', 'AverageBifurcationAngleLocal',
    'AverageBifurcationAngleRemote', 'HausdorffDimension',
    'pc11', 'pc12', 'pc13', 'pca_vr1', 'pca_vr2', 'pca_vr3'
]

BS7_COLORS = {
    'CTX': 'limegreen',
    'CNU': 'darkorange',
    'CB': 'royalblue',
    'TH': 'violet',
    'MB': 'sienna',
    'HY': 'mediumslateblue',
    'HB': 'red'
}


def load_features(mefile, scale=25., feat_type='mRMR', flipLR=True, standardize=True):
    df = pd.read_csv(mefile, index_col=0)

    if feat_type == 'full':
        cols = df.columns
        fnames = [fname for fname in cols if fname[-3:] == '_me']
    elif feat_type == 'mRMR':
        # Features selected by mRMR
        fnames = mRMR_f3me
    elif feat_type == 'PCA':
        fnames = ['pca_feat1', 'pca_feat2', 'pca_feat3']
    elif feat_type == 'single':
        fnames = mRMR_f3
    else:
        raise ValueError("Unsupported feature types")

    if standardize:
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
        nzi = np.nonzero(df['soma_z'] < zdim2)
        loci = df.index[nzi]
        df.loc[loci, 'soma_z'] = zdim - df.loc[loci, 'soma_z']

    return df, fnames

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

def moranI_score(coords, feats, eval_ids=None, reduce_type='average', threshold=0.5):
    """
    The coordinates should be in `mm`, and as type of numpy.array
    The feats should be standardized
    """
    # spatial coherence
    weights = pslib.weights.DistanceBand.from_array(coords, threshold=threshold)
    avgI = []
    if eval_ids is None:
        eval_ids = range(feats.shape[1])
    for i in eval_ids:
        moran = Moran(feats[:,i], weights)
        avgI.append(moran.I)
    
    if reduce_type == 'average':
        avgI = np.mean(avgI)
    elif reduce_type == 'max':
        avgI = np.max(avgI)
    elif reduce_type == 'all':
        return avgI
    else:
        raise NotImplementedError
    return avgI

def get_me_ccf_mapper(me2ccf_file):
    # load the me to ccf correspondence file
    with open(me2ccf_file, 'rb') as fm2c:
        me2ccf = pickle.load(fm2c)
    # get the reverse map
    ccf2me = {}
    for k, v in me2ccf.items():
        if v in ccf2me:
            ccf2me[v].append(k)
        else:
            ccf2me[v] = [k]

    return me2ccf, ccf2me

