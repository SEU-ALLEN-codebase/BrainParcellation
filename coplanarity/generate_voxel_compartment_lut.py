import numpy as np
import skimage.morphology
from scipy.ndimage import binary_dilation
from math import *
from utils import *
import os
import glob
import SimpleITK as sitk
from copy import deepcopy
from skimage import transform
from sklearn.neighbors import BallTree, KDTree
import configparser
# from visualization import triview

config = configparser.ConfigParser()
config.read('./coplanarity_config.ini')
__RES__ = float(config.get('GlobalParameter', 'resolution'))  # μm
__RADIUS__ = float(config.get('GlobalParameter', 'radius'))  # μm

import torch
from multiprocessing import Pool, cpu_count


def generate_anno(fp: str):
    annotmp = sitk.GetArrayFromImage(sitk.ReadImage(fp))
    anno = np.transpose(annotmp, axes=(2, 1, 0))
    anno[:, :, int(anno.shape[2] / 2):] = 0
    if __RES__ != 25:
        dst_shape = np.rint(np.array(anno.shape) * (25.0 / __RES__)).astype(int)
        anno = transform.resize(anno, dst_shape, order=0)  # nearest interpolation

    return anno


class VoxelCompartmentLut(object):
    def __init__(self, brain_region: str, anno: np.ndarray):
        self.brain_region = brain_region
        self.anno_mask = None
        self.voxel_array = None
        self.voxel_array_gpu = None
        self.qtree = None
        self.compartments = None
        self.v_c_lut = None
        self.f_matrix = None

        # for distance normalization (to 0-1)
        self.compartment_len_max = None
        self.compartment_accum_max = None

        self.initialize(anno)

    def initialize(self, anno: np.ndarray):
        self.generate_specific_anno(anno)

        return

    def _q_nodes_dim_check(self, q_nodes):
        q_nodes = np.asarray(q_nodes)
        if q_nodes.ndim == 1:
            q_nodes = q_nodes[np.newaxis, :]
        if q_nodes.ndim != 2 or q_nodes.shape[1] != 9:
            raise ValueError(f'"q_nodes" must be in shape (n, 9), received {q_nodes.shape}')

        return q_nodes

    def generate_specific_anno(self, anno: np.ndarray):
        mat = MouseAnatomyTree()
        br_id = mat.lutnametoid[self.brain_region]
        ch_id_list = mat.find_children_id(br_id)

        self.anno_mask = np.zeros(anno.shape, dtype=bool)
        cond = np.isin(anno, ch_id_list)
        self.anno_mask[cond] = True

        self.voxel_array = np.argwhere(cond)
        # self.voxel_array_gpu = torch.from_numpy(self.voxel_array).to('cuda')

        return

    def filter_qnodes_and_normalize(self, q_nodes):
        '''
        filter nodes using brain region mask
        :return:
        '''
        radius = __RADIUS__ / __RES__
        q_nodes = np.asarray(q_nodes)
        q_nodes = self._q_nodes_dim_check(q_nodes)

        xmin, ymin, zmin = np.min(self.voxel_array, axis=0) - radius
        xmax, ymax, zmax = np.max(self.voxel_array, axis=0) + radius
        cond_mask = (q_nodes[:, 0] >= xmin) & (q_nodes[:, 1] >= ymin) & (q_nodes[:, 2] >= zmin) & \
                    (q_nodes[:, 0] <= xmax) & (q_nodes[:, 1] <= ymax) & (q_nodes[:, 2] <= zmax)

        brain_region_mask = self.anno_mask[round(xmin):round(xmax + 1), round(ymin):round(ymax + 1),
                            round(zmin):round(zmax + 1)]

        q_nodes = q_nodes[cond_mask]
        q_nodes_pos_int = np.round(q_nodes[:, 0:3]).astype(int)

        footprint = skimage.morphology.ball(1)
        brain_region_mask_dilated = binary_dilation(brain_region_mask, footprint, iterations=ceil(radius))
        anno_mask_dilated = self.anno_mask.copy()
        anno_mask_dilated[round(xmin):round(xmax + 1), round(ymin):round(ymax + 1),
        round(zmin):round(zmax + 1)] = brain_region_mask_dilated

        x, y, z = q_nodes_pos_int.transpose()

        cond_mask = anno_mask_dilated[x, y, z]

        self.compartments = deepcopy(q_nodes[cond_mask])

        # normalization
        self.compartment_len_max = np.percentile(self.compartments[:, 6], 99)
        self.compartment_accum_max = np.percentile(self.compartments[:, 7], 99)

        self.compartments[:, 6] = np.exp(
            -np.clip(self.compartments[:, 6] / self.compartment_len_max, a_min=None, a_max=1))
        self.compartments[:, 7] = np.clip(self.compartments[:, 7] / self.compartment_accum_max, a_min=None, a_max=1)

        return

    # def filter_qnodes(self, q_nodes):
    #     '''
    #     filter nodes using bounding box of brain region
    #     :return:
    #     '''
    #     radius = __RADIUS__ / __RES__
    #     q_nodes = np.asarray(q_nodes)
    #     q_nodes = self._q_nodes_dim_check(q_nodes)
    #     xmin, ymin, zmin = np.min(self.voxel_array, axis=0) - radius
    #     xmax, ymax, zmax = np.max(self.voxel_array, axis=0) + radius
    #     # print((xmin, xmax), (ymin, ymax), (zmin, zmax))
    #     cond_mask = (q_nodes[:, 0] >= xmin) & (q_nodes[:, 1] >= ymin) & (q_nodes[:, 2] >= zmin) & \
    #                 (q_nodes[:, 0] <= xmax) & (q_nodes[:, 1] <= ymax) & (q_nodes[:, 2] <= zmax)
    #
    #     new_q_nodes = deepcopy(q_nodes[cond_mask])
    #
    #     return new_q_nodes

    def generate_query_tree(self, tree_type: str = 'balltree'):
        qTree = None
        if tree_type.lower() == 'balltree':
            qTree = BallTree
        elif tree_type.lower() == 'kdtree':
            qTree = KDTree
        self.qtree = qTree(self.compartments[:, 0:3], metric='minkowski', p=2)

        return

    def generate_lut(self):
        radius = __RADIUS__ / __RES__

        self.v_c_lut = self.qtree.query_radius(self.voxel_array, r=radius, return_distance=False)

        return

    def generate_fused_direction_vector(self, idx: int, ):
        tmp_voxel, tmp_compartments = self.voxel_array[idx], self.compartments[self.v_c_lut[idx]]
        # please note, xmin=0 because of flipping, ymin and zmin = -1
        tmp_fusion_d_v = np.array([0, 0, 0], dtype=np.float32)

        tmp_compartments_len = len(tmp_compartments)
        if tmp_compartments_len <= 100: return tmp_fusion_d_v
        tmp_pos = tmp_compartments[:, 0:3]
        tmp_d_v = tmp_compartments[:, 3:6]
        tmp_len = tmp_compartments[:, 6:7]
        tmp_accum_path = tmp_compartments[:, 7:8]
        tmp_vc_dist = np.linalg.norm(tmp_pos - tmp_voxel[np.newaxis, :], axis=1, keepdims=True).astype(np.float32)
        tmp_vc_dist = np.exp(-tmp_vc_dist * (__RES__ / __RADIUS__))

        # len_max = np.percentile(tmp_len, 99)
        # tmp_len = np.clip(tmp_len / len_max, a_min=None, a_max=1)
        # accum_path_max = np.percentile(tmp_accum_path, 99)
        # tmp_accum_path = np.clip(tmp_accum_path / accum_path_max, a_min=None, a_max=1)

        # todo: aggregate compartments and generate one direction vector,
        #  considering their length, accumulated path length, distance between compartment and voxel.
        #  now it is wrong.
        return
        tmp_fusion_d_v = tmp_d_v * tmp_len * tmp_accum_path * tmp_vc_dist
        tmp_fusion_d_v_sum = np.sum(tmp_fusion_d_v, axis=0)
        if tmp_fusion_d_v_sum[0] < 0:
            tmp_fusion_d_v_sum = -tmp_fusion_d_v_sum

        tmp_fusion_d_v_invx_sum = tmp_fusion_d_v.copy()
        cond = tmp_fusion_d_v[:, 0] < 0
        tmp_fusion_d_v_invx_sum[cond] = -tmp_fusion_d_v_invx_sum[cond]
        tmp_fusion_d_v_invy_sum = tmp_fusion_d_v.copy()
        cond = tmp_fusion_d_v[:, 1] < 0
        tmp_fusion_d_v_invy_sum[cond] = -tmp_fusion_d_v_invy_sum[cond]
        tmp_fusion_d_v_invz_sum = tmp_fusion_d_v.copy()
        cond = tmp_fusion_d_v[:, 2] < 0
        tmp_fusion_d_v_invz_sum[cond] = -tmp_fusion_d_v_invz_sum[cond]

        return tmp_fusion_d_v_abs

    def generate_feature_matrix(self):

        self.f_matrix = np.zeros((len(self.voxel_array), 3),
                                 dtype=np.float32)  # voxels*3 represent the direction vector of each voxel

        for i in range(len(self.voxel_array)):
            self.f_matrix[i] = self.generate_fused_direction_vector(i)
        return


def generate_vcl(all_compartments, brain_region, anno) -> VoxelCompartmentLut:
    t0 = time.time()

    vcl = VoxelCompartmentLut(brain_region, anno)

    t1 = time.time()
    print(f'initialize vcl: {t1 - t0:.3f}')

    all_compartments = compa_preprocess(all_compartments, __RES__)

    t2 = time.time()
    print(f'preprocess compartments: {t2 - t1:.3f}')

    vcl.filter_qnodes_and_normalize(all_compartments)

    t3 = time.time()
    print(f'filter compartments: {t3 - t2:.3f}', 'filtered compartments array shape: ', vcl.compartments.shape)

    vcl.generate_query_tree()

    t4 = time.time()
    print(f'initialize query tree: {t4 - t3:.3f}')

    vcl.generate_lut()

    t5 = time.time()
    print(f'query (assign lut): {t5 - t4:.3f}')

    # vcl.generate_feature_matrix()
    # #
    # t6 = time.time()
    # print(f'generate_feature_matrix: {t6 - t5:.3f}')

    # print(f'initialize vcl: {t1 - t0}\n'
    #       f'preprocess compartments: {t2 - t1}\n'
    #       f'filter compartments: {t3 - t2}\n'
    #       f'initialize query tree: {t4 - t3}\n'
    #       f'query (assign lut): {t5 - t4}'
    #       f'generate_feature_matrix: {t6 - t5}')
    return vcl


def compa_preprocess(compa: np.ndarray, resolution: float, gpu=False):
    '''
    rescale the positions and path distance (col index 0,1,2,6,7)
    flip direction vector (col index 3,4,5) to same hemisphere (x>0) (col index 3)
    :param resolution:
    :return:
    '''
    compa = np.asarray(compa)

    if gpu:
        compa = torch.from_numpy(compa).to('cuda')

    col_indice = np.r_[0:3, 6, 7]
    compa[:, col_indice] = compa[:, col_indice] * (1 / resolution)

    # cannot flip!!!
    # cond = compa[:, 3] < 0
    # compa[cond, 3:6] = -compa[cond, 3:6]

    if gpu:
        compa = compa.cpu().numpy()

    return compa


if __name__ == '__main__':
    import time

    anno = generate_anno(
        r'E:\ZhixiYun\Projects\GitHub\neuro_morpho_toolbox\neuro_morpho_toolbox\data\annotation_25.nrrd')
    all_compartments = np.load(r'G:\data\parcellation\binary\all_compartments.npy')
    print('all compartments array shape: ', all_compartments.shape)
    vcl = generate_vcl(all_compartments, 'CP', anno)
    print('brain region voxel size: ', len(vcl.voxel_array))

    # todo: save for visualzation of fused direction vector
    # # visualization
    #
    # alphalist = np.linalg.norm(vcl.f_matrix, axis=1, keepdims=True)
    # alphalist_eps = alphalist.copy()
    # alphalist_eps[alphalist_eps == 0] += 1e-10
    #
    # # f_matrix_unit = vcl.f_matrix / alphalist_eps  # wrong!!!
    # # print(np.max(f_matrix_unit, axis=0), np.min(f_matrix_unit, axis=0), np.median(f_matrix_unit, axis=0))
    # f_matrix_norm = vcl.f_matrix / alphalist_eps  # wrong!
    # f_matrix_unit[:, 1:3] = (f_matrix_unit[:, 1:3] + 1) / 2
    #
    # rgb_direction_anno = np.zeros(list(anno.shape) + [4], dtype=np.uint8)
    # x_idx, y_idx, z_idx = vcl.voxel_array[:, 0], vcl.voxel_array[:, 1], vcl.voxel_array[:, 2]
    # rgb_direction_anno[x_idx, y_idx, z_idx] = np.rint(
    #     np.hstack(
    #         [f_matrix_norm,
    #          np.clip(alphalist / np.percentile(alphalist, 90), a_min=None, a_max=1)]
    #     ) * 255.0).astype(np.uint8)
    #
    # sitk.WriteImage(sitk.GetImageFromArray(rgb_direction_anno), r'./tmp_rgb_direction_anno.nrrd')
