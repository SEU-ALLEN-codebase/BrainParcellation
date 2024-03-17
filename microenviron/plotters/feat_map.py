##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-14
#Description:               
##########################################################
import sys
import random
import pickle
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

sys.path.append('..')
from parcellation import load_features

from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE

def estimate_fmap_similarities(mefile, dist='pearson'):
    # randomly select 10K samples for estimation
    np.random.seed(1024)
    random.seed(1024)
    ntotal = 103603 # number of samples
    nsel = 1000
    sindices = random.sample(np.arange(ntotal).tolist(), nsel)


    # utility
    def get_data(mefile, feat_type, dist, normalize=True):
        print(f'Loading {feat_type} features')
        dff, fnames = load_features(mefile, feat_type=feat_type); dff = dff[fnames]    
        
        dffs = dff.iloc[sindices]
        if dist == 'pearson':
            # pairwise correlations
            dmatrix = dffs.transpose().corr().values
            # select all pairs
            triui = np.triu_indices(nsel, k=1)
            values = dmatrix[triui[0],triui[1]]
        elif (dist == 'cityblock') or (dist == 'euclidean'):
            #import ipdb; ipdb.set_trace()
            values = pdist(dffs, metric=dist)

        print(values.mean(), values.std())
        if normalize:
            values = (values - values.mean()) / values.std()

        return values
    
    full = get_data(mefile, 'full', dist, True)
    mrmr = get_data(mefile, 'mRMR', dist, True)
    pca = get_data(mefile, 'PCA', dist, True)

    pfm = pearsonr(full, mrmr)
    pfp = pearsonr(full, pca)
    pmp = pearsonr(mrmr, pca)
    print(f'[full-mrmr]: {pfm.correlation:.4f}')
    print(f'[full-pca]: {pfp.correlation:.4f}')
    print(f'[mrmr-pca]: {pmp.correlation:.4f}')

    # plot 
    # there are two many points, and it is hard to display it at once. 
    # We randomly select a subset of it for the visualization purpose
    ndisp = 5000
    data = np.stack((full, mrmr, pca)).transpose()
    dindices = random.sample(np.arange(full.shape[0]).tolist(), ndisp)
    data = data[dindices]
    df = pd.DataFrame(data, columns=('full', 'mRMR', 'PCA'))
    
    sns.pairplot(df, plot_kws={'marker':'.', 'color':'orange', 'line_kws':dict(color='m')}, kind='reg')
    plt.savefig('feature_correlations.png', dpi=300)
    plt.close()


def get_correspondence_map(parc1, parc2, rmask):
    # estimate the correspondence of the sub-regions of parc2 to parc1
    regs1, cnts1 = np.unique(parc1[rmask], return_counts=True)
    regs2, cnts2 = np.unique(parc2[rmask], return_counts=True)
    # initialize the correspondence matrix
    rsize1, rsize2 = regs1.size, regs2.size
    cmat = np.zeros((rsize1, rsize2))
    for irid2, rid2 in enumerate(regs2):
        # find its correspondence in parc1
        mask2 = parc2 == rid2
        regs21 = parc1[mask2]
        cmat[:,irid2] = np.histogram(regs21, bins=rsize1, range=(1,rsize1+1))[0]
    # normalize
    cmat /= (cmat.sum(axis=0).reshape(1,-1) + 1e-10)

    return cmat

def estimate_parc_similarities(parc_files):

    # utilities
    def get_customized_cmap():
        # modified according to: https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
        cvals = [0,1]
        colors = ["white", "red"]
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
        
        return cmap

    def parc_correspondence(parc1, parc2, rmask, prefix='temp'):
        cmat = get_correspondence_map(parc1, parc2, rmask)
        rsize1, rsize2 = cmat.shape
        df = pd.DataFrame(cmat, index=np.arange(rsize1)+1, columns=np.arange(rsize2)+1)
        print(df)
        cmap = get_customized_cmap()
        # annot the max value for each region
        cargmax = cmat.argmax(axis=0)
        annot = np.zeros_like(cmat).astype(str)
        annot[:] = ''
        vmaxs = np.round(cmat[cargmax, np.arange(cmat.shape[1])], 2).astype(str)
        annot[cargmax, np.arange(cmat.shape[1])] = vmaxs
        sns.heatmap(df, cmap=cmap, annot=annot, fmt='s', annot_kws={'c':'k', 'fontsize':5, 'fontweight':'bold'})
        plt.title(prefix)
        plt.gca().spines[:].set_visible(True)
        plt.savefig(f'parc_correspondence_CP_{prefix}.png', dpi=300)
        plt.close()

    print('Loading the parcellation using different sets of features...')
    full = load_image(parc_files['full'])
    mrmr = load_image(parc_files['mRMR'])
    pca = load_image(parc_files['PCA'])
    assert((full>0).sum() == (mrmr>0).sum() and (full>0).sum() == (pca>0).sum())

    rmask = full > 0
    print('Now we will compare there difference')
    parc_correspondence(mrmr, pca, rmask, 'mRMR-PCA')
    parc_correspondence(mrmr, full, rmask, 'mRMR-full')
    print()

def find_best_feat_type(rmap_file, parc_file, r314_mask_file, r671_mask_file, flipLR=True):
    """We can choose the best feature type based on the separability of well-defined CCF regions"""
    # load the region mapping file
    with open(rmap_file, 'rb') as fp:
        rmap = pickle.load(fp)[0]
    # load the CCFv3 masks
    r314_mask = load_image(r314_mask_file)
    r671_mask = load_image(r671_mask_file)
    if flipLR:
        zs = r314_mask.shape[0]//2
        r314_mask[:zs] = 0
        r671_mask[:zs] = 0
    # load the parcellation
    parc = load_image(parc_file)

    # Then we find out the hierarchical regions
    for r1, rs in rmap.items():
        if len(rs) > 1:
            rmask = r314_mask == r1
            cmat = get_correspondence_map(r671_mask, parc, rmask)
            prids = np.unique(parc[rmask])
            print(r1, len(rs), prids)
            # check no errors
            assert(rmask.sum() == (r671_mask[rmask]>0).sum())
            assert(rmask.sum() == (parc[rmask]>0).sum())
            
            
            #import ipdb; ipdb.set_trace()
            print()
    
    

if __name__ == '__main__':
    if 0:
        mefile = '../data/mefeatures_100K_with_PCAfeatures3.csv'
        dist = 'euclidean'
        estimate_fmap_similarities(mefile, dist=dist)

    if 0:
        parc_files = {
            'full': '../intermediate_data/parc_full_region672.nrrd',
            'mRMR': '../intermediate_data/parc_mRMR_region672.nrrd',
            'PCA': '../intermediate_data/parc_PCA_region672.nrrd'
        }
        estimate_parc_similarities(parc_files)

    if 1:
        rmap_file = '/home/lyf/Softwares/installation/pylib/anatomy/resources/region671_to_region314_woFiberTracts.pkl'
        parc_file = '../intermediate_data/parc_r671_mrmr.nrrd'
        r314_mask_file = MASK_CCF25_R314_FILE
        r671_mask_file = MASK_CCF25_FILE
        find_best_feat_type(rmap_file, parc_file, r314_mask_file, r671_mask_file)


