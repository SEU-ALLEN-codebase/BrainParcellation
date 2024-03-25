##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-14
#Description:               
##########################################################
import os
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
from utils import crop_regional_mask

from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE, MASK_CCF25_R314_FILE

def estimate_fmap_similarities(mefile, dist='pearson'):
    # randomly select 10K samples for estimation
    np.random.seed(1024)
    random.seed(1024)
    ntotal = 103603 # number of samples
    nsel = 1000
    sindices = random.sample(np.arange(ntotal).tolist(), nsel)

    # configuring figures
    sns.set_context("paper", rc={"axes.labelsize":18})


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
    
    sns.pairplot(df, plot_kws={'marker':'.', 'color':'cornflowerblue', 'line_kws':dict(color='red')}, 
                 diag_kws={'color':'darkgray'}, kind='reg')
    plt.savefig('feature_correlations.png', dpi=300)
    plt.close()


def get_correspondence_map(parc1, parc2, rmask):
    # estimate the correspondence of the sub-regions of parc2 to parc1
    regs1, cnts1 = np.unique(parc1[rmask], return_counts=True)
    regs2, cnts2 = np.unique(parc2[rmask], return_counts=True)
    # initialize the correspondence matrix
    rsize1, rsize2 = regs1.size, regs2.size
    cmat = pd.DataFrame(np.zeros((rsize1, rsize2)), columns=regs2, index=regs1)
    for irid2, rid2 in enumerate(regs2):
        # find its correspondence in parc1
        mask2 = parc2 == rid2
        regs21 = parc1[mask2]
        #cmat[:,irid2] = np.histogram(regs21, bins=rsize1, range=(1,rsize1+1))[0]
        univ = np.unique(regs21, return_counts=True)
        cmat.loc[univ[0], rid2] = univ[1]
        
    # normalize
    cmat = cmat.values
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
        g = sns.heatmap(df, cmap=cmap, annot=annot, fmt='s', annot_kws={'c':'k', 'fontsize':5, 'fontweight':'bold'},
                    xticklabels=1, yticklabels=1, cbar_kws={'label':'Overlap ratio', 'pad':0.02})
        label_size = 15
        #plt.title(prefix, fontsize=18)
        plt.ylabel('Sub-region ID (full)', fontsize=label_size)
        xlabel = prefix.split("-")[1]
        plt.xlabel(f'Sub-region ID ({xlabel})', fontsize=label_size)
        
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
    parc_correspondence(full, pca, rmask, 'full-PCA')
    parc_correspondence(full, mrmr, rmask, 'full-mrmr')
    print()

def find_best_feat_type(rmap_file, parc_files, r314_mask_file, r671_mask_file, 
                        flipLR=True, same_subregions_only=True, plot=True):
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
    parcs = []
    for parc_file in parc_files:
        parc = load_image(parc_file)
        parcs.append(parc)
        print(f'{os.path.split(parc_file)[-1]}: {parc.max()}')

    # Then we find out the hierarchical regions
    scores = []
    all_scores = [[] for i in range(len(parcs))]
    for r1, rs in rmap.items():
        if len(rs) > 1:
            # to speed up, use cropped mask
            r314ms,zmin,zmax,ymin,ymax,xmin,xmax = crop_regional_mask(r314_mask, v=r1)
            r671ms = r671_mask[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]

            rmask = r314ms == r1
            cnt314 = rmask.sum()
            cnt671 = (r671ms[rmask]>0).sum()
            assert(cnt314 == cnt671)
            
            nregions = []
            for parc in parcs:
                sub_parc = parc[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1]
                assert(cnt314 == (sub_parc[rmask]>0).sum())
                prids = np.unique(sub_parc[rmask])
                nregions.append(len(prids))
            print(r1, len(rs), nregions)
            
            # we only estimate based on the regions with the same sub-regions
            if same_subregions_only and (len(np.unique(nregions)) != 1):
                continue
            else:
                cur_scores = []
                for i, parc in enumerate(parcs):
                    cmat = get_correspondence_map(r671ms, parc[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1], rmask)
                    score = cmat.max(axis=0)
                    avg_score = score.mean()
                    cur_scores.append(avg_score)
                    all_scores[i].extend(score.tolist())
                scores.append(cur_scores)
                print(f'==> {cur_scores}')
    
    scores = np.array(scores)
    print(scores.shape)
    print(scores.mean(axis=0), scores.std(axis=0)); print()

    if plot:
        for i, parc_file in enumerate(parc_files):
            scores_i = all_scores[i]
            parc_name = os.path.split(parc_file)[-1][:-5]
            df = pd.DataFrame(scores_i, columns=('score',))
            fig, ax = plt.subplots(1,1,figsize=(8,8))
            sns.histplot(df, x='score', color='silver', edgecolor='black', bins=20, stat='proportion', legend=False)
            plt.xlim(0, 1.01)
            plt.xlabel('Score')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            plt.savefig(f'{parc_name}.png', dpi=300)
            plt.close()
        print()
    
    

if __name__ == '__main__':
    # calculate the similarity between different sets of features
    if 0:
        mefile = '../data/mefeatures_100K_with_PCAfeatures3.csv'
        dist = 'euclidean'
        estimate_fmap_similarities(mefile, dist=dist)

    # calculate the parcellation similarity based on different feature types
    if 1:
        parc_files = {
            'full': '../intermediate_data/parc_full_region672.nrrd',
            'mRMR': '../intermediate_data/parc_mRMR_region672.nrrd',
            'PCA': '../intermediate_data/parc_PCA_region672.nrrd'
        }
        estimate_parc_similarities(parc_files)

    # compare the parcellation with CCF anatomy
    if 0:
        rmap_file = '/home/lyf/Softwares/installation/pylib/anatomy/resources/region671_to_region314_woFiberTracts.pkl'
        parc_files = ['../intermediate_data/parc_r314_mrmr.nrrd', 
                     '../intermediate_data/parc_r314_pca.nrrd', 
                     '../intermediate_data/parc_r314_full.nrrd', ]
        same_subregions_only = False
        r314_mask_file = MASK_CCF25_R314_FILE
        r671_mask_file = MASK_CCF25_FILE
        find_best_feat_type(rmap_file, parc_files, r314_mask_file, r671_mask_file, same_subregions_only=same_subregions_only)


