##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-15
#Description:               
##########################################################
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_score(conf_matrix):
    ix, iy = np.triu_indices_from(conf_matrix,k=1)
    avg_offdiag = conf_matrix[ix,iy].mean()
    avg_diag = np.diagonal(conf_matrix).mean()
    # normalized score
    score = avg_offdiag / avg_diag
    return score

def merfish_score(prm_merfish):
    df = pd.read_csv(prm_merfish, index_col=0)
    clusters = df.leiden_label
    coords = df[['x', 'y', 'z']]
    uniq_c = np.unique(clusters)
    uniq_cl = len(uniq_c)

    pdists = cdist(coords, coords, metric='euclidean')
    dmatrix = np.zeros((uniq_cl, uniq_cl))
    for i in range(uniq_cl):
        for j in range(i, uniq_cl):
            pds = pdists[clusters==i, :][:, clusters==j]
            dmean = pds.mean()
            dmatrix[i,j] = dmean
            dmatrix[j,i] = dmean
            
    # post-process the matrix
    M = (dmatrix.max() - dmatrix) / (dmatrix.max() - dmatrix.min())
    
    import ipdb; ipdb.set_trace()
    print()

if __name__ == '__main__':
    merfish_file = './data/prm_g_org.csv'
    merfish_score(merfish_file)
    
