##########################################################
#Author:          Yufeng Liu
#Create time:     2025-04-07
#Description:               
##########################################################
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

import seaborn as sns
import matplotlib.pyplot as plt

from anatomy.anatomy_config import SALIENT_REGIONS
from ml.feature_processing import standardize_features

import sys
sys.path.append('..')
sys.path.append('../../common_lib')
from configs import __FEAT_NAMES__
from config import load_features, __FEAT24D__


def processing_data(df_in, scale=25., flipLR=True, standardize=True, is_me=True):
    df = df_in.copy()
    if is_me:
        fnames = [fname for fname in df.columns if fname.endswith('_me')]
        standardize_features(df, __FEAT24D__+fnames, inplace=True)
    else:
        fnames = __FEAT24D__
        standardize_features(df, __FEAT24D__, inplace=True)

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
        # keep only neurons in salient regions
        df_s = df_s[df_s.region_id_r671.isin(SALIENT_REGIONS)]

        data[metype] = df_s

        print(metype, df_s.shape)
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

            if metype in ('variance-normalized-median', 'median-std-stdn', 'median-std', 
                          'mean', 'mean_all', 'median', 'median_all'):
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
        # show only the top-5 components
        df = df.iloc[:, :5]

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
        # rename
        sns.lineplot(data=melted_df.replace('spatial-weighting', 'ME').replace('dendrite-only', 'Single-neuron dendrite'), x='dimension', y='vratio', 
                     hue='method', marker='o', lw=2, markersize=12,
                     alpha=1.0)
        plt.title('Cumulative variance comparison across methods')
        plt.xlim(0.5, 5.5)
        plt.ylim(5, 18)
        plt.xlabel('Number of top-ranking PCA components')
        plt.ylabel('Cumulative variance')
        #plt.xticks(range(1, 25, 2))
        #plt.grid(True, linestyle='--')
        #plt.legend(ncol=2, frameon=False, markerscale=3)
        plt.legend(labelspacing=0.2, handletextpad=0.2,
                       borderpad=0.05, frameon=False,
                       fontsize=16, alignment='center', ncols=2,
                       markerscale=1.25, columnspacing=0.8)

        ax = plt.gca()
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        plt.tight_layout()
        plt.savefig(f'cumulative_variances_various_methods.png', dpi=300)
        plt.close()
        print()

    
    if 0:
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


    if 1:

        ############### Helper functions ##################
        def evaluate_clustering(df, fnames, max_k=10):
            """
            参数：
            df: 输入DataFrame
            fnames: 用于聚类的特征列名列表
            max_k: 最大聚类数（默认到10）
            
            返回：
            results: 包含k值和对应silhouette score的DataFrame
            """
            # 1. 数据预处理
            X = df[fnames].values
            
            # 2. 遍历k值并计算silhouette score
            results = []
            for k in range(2, max_k+1):
                #print(k)
                cluster_ = KMeans(n_clusters=k, random_state=1024, n_init='auto')
                #cluster_ = SpectralClustering(n_clusters=k, random_state=1024)
                cluster_labels = cluster_.fit_predict(X)
                
                # 计算silhouette score（忽略单聚类情况）
                if len(np.unique(cluster_labels)) >= 2:
                    score = silhouette_score(X, cluster_labels)
                else:
                    score = np.nan
                    
                results.append({'k': k, 'silhouette_score': score})
            
            # 3. 转换为DataFrame并可视化
            results_df = pd.DataFrame(results)

            return results_df
        ############ End of helper functions ##############


        # Clustering and evaluate the silhoutte scores
        imethods = ('spatial-weighting', 'mean', 'mean_all', 'median', 'median_all')
        
        regions = data[imethods[0]].region_name_r316
        min_neurons = 20
        max_clusters = 5

        scores_file = 'sil_scores_across_methods.csv'
        if os.path.exists(scores_file):
            df_scores = pd.read_csv(scores_file, index_col=0)
        
        else:
            regions_set, regions_counts = np.unique(regions, return_counts=True)
            regions_kept = regions_set[regions_counts > min_neurons]

            scores = {'dendrite-only': []}
            for imethod in imethods:
                scores[imethod] = []

            im = 0
            for imethod in imethods:
                df = data[imethod]
                fnames = [fname for fname in df.columns if fname.endswith('_me')]
                
                for region in regions_kept:
                    # do clustering
                    df_reg = df[df.region_name_r316 == region]
                    sil_score = evaluate_clustering(df_reg, fnames, max_clusters)
                    scores[imethod].append(sil_score.silhouette_score.max())

                    if im == 0:
                        sil_score_orig = evaluate_clustering(df_reg, __FEAT24D__, max_clusters)
                        scores['dendrite-only'].append(sil_score_orig.silhouette_score.max())

                    print(f'[{imethod}/{region}]: {sil_score.silhouette_score.max():.3f}')

                im += 1

            df_scores = pd.DataFrame(scores)
            df_scores['region'] = regions_kept
            df_scores.to_csv(scores_file, float_format='%.4f')
        
        
        # filter out non-standard regions
        df_scores = df_scores[~(df_scores.region.isin(['error', 'fiber tracts']))]
        
        # plotting
        sns.set_theme(style='ticks', font_scale=1.8)

        baseline = 'dendrite-only'
        methods_to_compare = ['spatial-weighting', 'mean_all']

        # 统计每个方法比baseline高的区域数
        improved_counts = {}
        for method in methods_to_compare:
            improved_counts[method] = (df_scores[method] > df_scores[baseline]).sum()

        # 转换为DataFrame便于绘图
        result_df = pd.DataFrame.from_dict(improved_counts, orient='index', columns=['Count'])
        result_df = result_df.reset_index().rename(columns={'index': 'Method'})

        # 绘制条形图
        plt.figure(figsize=(8, 6))
        bars = plt.bar(result_df['Method'], result_df['Count'], color=['skyblue', 'salmon'], width=0.4)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}',
                     ha='center', va='bottom')

        # 计算baseline总区域数（排除error和fiber tracts等非标准区域）
        total_regions = len(df_scores)

        # 添加参考线和说明
        plt.axhline(y=total_regions, color='gray', linestyle='--', alpha=0.5)
        plt.text(0.5, total_regions-10, f'Total regions: {total_regions}', ha='center', color='gray')

        plt.title('Number of Regions with Improved Silhouette Scores\n(Compared to dendrite-only baseline)')
        plt.ylabel('Number of Regions')
        plt.ylim(0, total_regions + 10)
        plt.xlim(-0.5, 1.5)
        #plt.grid(axis='y', alpha=0.3)
        plt.savefig('improved_scores.png', dpi=300)
        plt.close()

        # 输出详细统计结果
        print("Detailed comparison:")
        for method in methods_to_compare:
            improved_regions = df_scores[df_scores[method] > df_scores[baseline]]['region'].tolist()
            print(f"\n{method} outperforms baseline in {len(improved_regions)} regions:")
            print(improved_regions)

        print()
        


if __name__ == '__main__':
    file_dict = {
        'spatial-weighting': '../data/mefeatures_100K_with_PCAfeatures3.csv',
        #'mean': '../data/mefeatures_100K_mean.csv',
        #'mean_all': '../data/mefeatures_100K_mean_all.csv',
        #'median': '../data/mefeatures_100K_median.csv',
        #'median_all': '../data/mefeatures_100K_median_all.csv',
        #'median-std': '../data/mefeatures_100K_median-std.csv',
        #'median-std-stdn': '../data/mefeatures_100K_median-std-stdn.csv',
        #'variance-normalized-median': '../data/mefeatures_100K_variance-normalized-median.csv',
    }
    cache_file = 'mefeatures_all_methods.pkl'

    #prepare_all_data(file_dict, cache_file=cache_file)
    calc_eigenvalues(cache_file)

