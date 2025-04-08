##########################################################
#Author:          Yufeng Liu
#Create time:     2025-04-07
#Description:               
##########################################################
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
sys.path.append('../../common_lib')
from configs import __FEAT_NAMES__
from config import load_features, __FEAT24D__


def processing_data(df_in, scale=25., flipLR=True, standardize=True):
    df = df_in.copy()
    fnames = __FEAT24D__ + [fname for fname in df.columns if fname.endswith('_me')]

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

def prepare_all_data(file_dict, cache_file):
    # load all files
    data = {}
    for metype, mefile in file_dict.items():
        # load the data
        df = pd.read_csv(mefile, index_col=0, low_memory=False)
        
        # find out the non-neighboring neurons
        me_feats = [col for col in df.columns if col.endswith('_me')]

        s_neurons = set(df.index[~(df[me_feats].isna().all(axis=1))])
        if len(data) == 0:
            skeys = s_neurons
        else:
            skeys = skeys & s_neurons

        print(f'Number of salient neurons: {len(skeys)}')

        data[metype] = df

    # remove neurons do not exist in all sets
    cnter = 0
    for metype, df in data.items():
        df_s = df[df.index.isin(skeys)]
        # processing
        df_s, _ = processing_data(df_s)
        data[metype] = df_s

        cnter += 1

    with open(cache_file, 'wb') as f1:
        pickle.dump(data, f1)

    return data
    

def calc_eigenvalues(data_file):

    ################ Helper functions ##############
    def get_vratio(feats):
        pca = PCA(n_components=feats.shape[1], whiten=False)
        feats_transformed = pca.fit_transform(feats)
        vratio = pca.explained_variance_#ratio_

        return vratio

    ############## End of helper functions #########


    with open(data_file, 'rb') as f1:
        data = pickle.load(f1)


    if 1:
        # Calculate the eigenvalues of PCA transformation
        vratio_dict = {}
        nbase = len(__FEAT24D__)
        for metype, df in data.items():
            if len(vratio_dict) == 0:
                # get the original dendrite-only features
                feats = df[__FEAT24D__]
                vratio_dict['dendrite-only'] = get_vratio(feats)

            if metype in ('variance-normalized-median', 'median-std-stdn'):
                continue
            
            fnames = [fname for fname in df.columns if fname.endswith('_me')]
            vratio = get_vratio(df[fnames])
            # scaling and extraction the first 24D
            vratio = vratio[:nbase]
            vratio_dict[metype] = vratio
            
        # reformat to dataframe
        df = pd.DataFrame.from_dict(vratio_dict, orient='index')

        # 重命名列
        df.columns = [f'v{i+1}' for i in range(nbase)]

        # 将索引转换为'method'列
        df = df.reset_index().rename(columns={'index': 'method'})
        cumsum_df = df.set_index('method').cumsum(axis=1)
        #cumsum_df = cumsum_df / cumsum_df.max(axis=1).values.reshape((-1,1))

        # plotting
        # 数据准备（假设df是之前转换的DataFrame）
        melted_df = cumsum_df.reset_index().melt(id_vars='method', var_name='dimension', value_name='vratio')
        melted_df['dimension'] = melted_df['dimension'].str.extract('(\d+)').astype(int)  # 提取v1/v2...中的数字

        # 绘制折线图
        sns.set_theme(style='ticks', font_scale=1.8)
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=melted_df, x='dimension', y='vratio', 
                     hue='method', marker='o', lw=2, markersize=8,
                     alpha=1.0)
        plt.title('Comparison of cumulative variance across Methods')
        plt.xlabel('PCA dimension')
        plt.ylabel('Cumulative variance')
        plt.xticks(range(1, 25, 2))
        #plt.grid(True, linestyle='--')
        #plt.legend(ncol=2, frameon=False, markerscale=3)
        plt.legend(labelspacing=0.1, handletextpad=0.1,
                       borderpad=0.05, frameon=False,
                       fontsize=16, alignment='center', ncols=2,
                       markerscale=1.8, columnspacing=0.5)

        ax = plt.gca()
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tight_layout()
        plt.savefig(f'cumulative_variances_various_methods.png', dpi=300)
        plt.close()
        print()

    
    if 1:
        # pairwise correlation-coefficients
        sns.set_theme(style='ticks', font_scale=1.8)
        
        df_mss = data['median-std']
        fnames = [fname for fname in df_mss.columns if fname.endswith('_me')]
        feats_mss = df_mss[fnames]

        # correlation
        corr_matrix = feats_mss.corr()

        # visualization
        plt.figure(figsize=(12, 10))
        g_corr = sns.heatmap(
            corr_matrix, 
            annot=False,          # 是否显示数值
            cmap='coolwarm',      # 颜色映射
            vmin=-1, vmax=1,      # 颜色范围（-1到1）
            center=0,             # 中心点为0
            linewidths=0.5,       # 单元格边线宽度
            square=True,           # 单元格为正方形
            cbar_kws={'shrink':0.2, 'aspect':6, 'anchor':(0,0)},
        )
        cbar = g_corr.collections[0].colorbar
        cbar.ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.tick_params(length=0, labelsize=12)

        plt.title('Feature Correlation Heatmap')

        # 
        g_corr.tick_params(
            axis='both',          # 同时控制x和y轴
            which='both',         # 同时控制major和minor ticks
            bottom=False,         # 移除x轴下方ticks
            left=False,           # 移除y轴左侧ticks
            #labelbottom=False,    # 移除x轴下方labels
            #labelleft=False       # 移除y轴左侧labels
        )

        plt.savefig('pairwise_feature_corr.png', dpi=300)
        plt.close()
        


if __name__ == '__main__':
    file_dict = {
        'spatial-weighting': '../data/mefeatures_100K_with_PCAfeatures3.csv',
        'mean': '../data/mefeatures_100K_mean.csv',
        'median': '../data/mefeatures_100K_median.csv',
        'median-std': '../data/mefeatures_100K_median-std.csv',
        'median-std-stdn': '../data/mefeatures_100K_median-std-stdn.csv',
        'variance-normalized-median': '../data/mefeatures_100K_variance-normalized-median.csv',
    }
    cache_file = 'mefeatures_all_methods.pkl'

    #prepare_all_data(file_dict, cache_file=cache_file)
    calc_eigenvalues(cache_file)

