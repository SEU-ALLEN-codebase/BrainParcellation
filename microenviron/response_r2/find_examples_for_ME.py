##########################################################
#Author:          Yufeng Liu
#Create time:     2025-04-12
#Description:               
##########################################################
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from ml.feature_processing import standardize_features
from plotters.neurite_arbors import NeuriteArbors

import sys
sys.path.append('..')
sys.path.append('../../common_lib')
from config import load_features, __FEAT24D__
from comparing_integrating_methods import processing_data


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_gf_diff_me(gf, me, fnames_gf, fnames_me, 
                          gf_similarity_threshold=0.9, 
                          me_difference_threshold=0.5):
    """
    找出GF特征相似但ME特征有差异的行
    
    参数:
    - gf: 包含GF特征的DataFrame
    - me: 包含ME特征的DataFrame
    - fnames_gf: GF特征列名列表
    - fnames_me: ME特征列名列表
    - gf_similarity_threshold: GF特征的相似度阈值(默认0.9)
    - me_difference_threshold: ME特征的差异阈值(默认0.5)
    
    返回:
    - 包含匹配行对的DataFrame,每对包含两行的索引和相似度/差异分数
    """
    
    # 提取GF和ME特征
    gf_features = gf[fnames_gf].values
    me_features = me[fnames_me].values
    
    # 计算GF特征之间的余弦相似度矩阵
    gf_sim = cosine_similarity(gf_features)
    
    # 计算ME特征之间的余弦相似度矩阵(用于差异计算)
    me_sim = cosine_similarity(me_features)
    
    # 获取上三角矩阵的索引(避免重复比较)
    rows, cols = np.triu_indices_from(gf_sim, k=1)

    # 向量化提取相似度和差异度
    gf_sim_values = gf_sim[rows, cols]
    me_diff_values = 1 - me_sim[rows, cols]
    
    # 创建条件掩码
    condition_mask = (gf_sim_values >= gf_similarity_threshold) & \
                    (me_diff_values >= me_difference_threshold)
    
    # 应用条件筛选
    matched_rows = rows[condition_mask]
    matched_cols = cols[condition_mask]
    gf_sim_results = gf_sim_values[condition_mask]
    me_diff_results = me_diff_values[condition_mask]
    
    # 构建结果DataFrame
    result_df = pd.DataFrame({
        'row_index_1': gf.index[matched_rows],
        'row_index_2': gf.index[matched_cols],
        'gf_similarity': gf_sim_results,
        'me_difference': me_diff_results
    })
    
    # 按差异度降序排序
    if not result_df.empty:
        result_df = result_df.sort_values('me_difference', ascending=False)
    
    return result_df


def find_neurons_within_radius(gf, me, fnames_gf, fnames_me, 
                             coordinate_cols=['soma_x', 'soma_y', 'soma_z'],
                             gf_similarity_threshold=0.9,
                             me_difference_threshold=0.5,
                             radius_um=166.36,
                             voxel_size=25):
    """
    1. 找出GF特征相似但ME特征有差异的神经元对
    2. 对每个满足条件的神经元，找出其周围radius_um范围内的其他神经元
    
    参数:
    - gf, me: 包含神经元特征的DataFrame
    - fnames_gf, fnames_me: 特征列名列表
    - coordinate_cols: 包含坐标的列名
    - gf_similarity_threshold: GF特征相似度阈值
    - me_difference_threshold: ME特征差异阈值
    - radius_um: 搜索半径(微米)
    - voxel_size: 体素大小(微米)，用于坐标转换
    
    返回:
    - 字典: {满足条件的神经元: [周围166.36μm内的神经元列表]}
    """
    
    # 步骤1: 找出GF相似但ME有差异的神经元对
    similar_pairs = find_similar_gf_diff_me(
        gf, me, fnames_gf, fnames_me,
        gf_similarity_threshold,
        me_difference_threshold
    )
    
    # 获取所有满足条件的唯一神经元(合并两个列)
    unique_neurons = pd.unique(
        np.concatenate([similar_pairs['row_index_1'], similar_pairs['row_index_2']])
    )
    
    # 步骤2: 为每个满足条件的神经元找出周围神经元
    
    # 确定坐标在哪个DataFrame中
    if all(col in gf.columns for col in coordinate_cols):
        coord_df = gf
    elif all(col in me.columns for col in coordinate_cols):
        coord_df = me
    else:
        raise ValueError("坐标列不在gf或me中")
    
    # 获取所有神经元的坐标(转换为实际微米坐标)
    all_coords = coord_df[coordinate_cols].values * voxel_size
    neuron_names = coord_df.index
    
    # 创建距离矩阵(单位:微米)
    distance_matrix = cdist(all_coords, all_coords, 'euclidean')
    
    # 为每个满足条件的神经元找出周围神经元
    result = {}
    for neuron in unique_neurons:
        # 找到该神经元在距离矩阵中的索引
        neuron_idx = np.where(neuron_names == neuron)[0][0]
        
        # 找出距离内的神经元(不包括自己)
        within_radius = (distance_matrix[neuron_idx] <= radius_um) & \
                       (distance_matrix[neuron_idx] > 0)
        
        # 获取神经元名称
        nearby_neurons = neuron_names[within_radius].tolist()
        
        if nearby_neurons:
            result[neuron] = nearby_neurons
    
    return similar_pairs, result
    
def find_distal_similar_neurons(gf_file, me_file, swc_dir, save_swc_image=True):
    gf = pd.read_csv(gf_file, index_col=0, low_memory=False)
    me = pd.read_csv(me_file, index_col=0, low_memory=False)
    # get the common neurons
    gf = gf.loc[me.index]

    # map the distance 
    gf, fnames_gf = processing_data(gf, is_me=False)
    me, fnames_me = processing_data(me, is_me=True) 

    # get the pairwise distances
    similar_pairs, neighboring_neurons = find_neurons_within_radius(gf, me, fnames_gf, fnames_me)

    if save_swc_image:
        ############### Helper functions ###################
        def _plot(swc_dir, swc, plot_type, pair_name, iswc, is_target=True):
            swcfile = os.path.join(swc_dir, f'{swc}.swc')
            na = NeuriteArbors(swcfile)
            if is_target:
                figname = f'{pair_name}-p{iswc}'
            else:
                figname = f'{pair_name}-p{iswc}-{swc}'

            na.plot_morph_mip(plot_type, color='b', figname=figname, out_dir='.', show_name=False)

        ####################################################



        plot_type = [3,4]
        for ipair, pair in similar_pairs.iterrows():
            print(ipair)
            swc1, swc2 = pair.row_index_1, pair.row_index_2
            neighbors1 = neighboring_neurons[swc1]
            neighbors2 = neighboring_neurons[swc2]
            neighbors = [neighbors1, neighbors2]
            # plotting
            pair_name = f'pair-{swc1}-{swc2}'
            for iswc, swc in enumerate([swc1, swc2]):
                _plot(swc_dir, swc, plot_type, pair_name, iswc, True)

                for nswc in neighbors[iswc]:
                    _plot(swc_dir, nswc, plot_type, pair_name, iswc, False)

    import ipdb; ipdb.set_trace()
    print()

if __name__ == '__main__':
    den_lm_file = '../plotters/hippocampus/ION_HIP/lm_features_d28_dendrites.csv'
    me_file = '../plotters/hippocampus/ION_HIP/mefeatures_dendrites.csv'
    swc_dir = '../plotters/hippocampus/ION_HIP/swc_dendrites'

    find_distal_similar_neurons(den_lm_file, me_file, swc_dir)
    
