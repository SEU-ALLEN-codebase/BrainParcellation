import numpy as np

from utils import *
import os
import glob
import SimpleITK as sitk
from copy import deepcopy
from skimage import transform
from sklearn.neighbors import BallTree, KDTree
import configparser

config = configparser.ConfigParser()
config.read('./coplanarity_config.ini')
__RES__ = float(config.get('GlobalParameter', 'resolution'))  # μm
__RADIUS__ = float(config.get('GlobalParameter', 'radius'))  # μm

import torch


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
        self.voxel_array = None
        self.qtree = None
        self.v_c_lut = []

        self.initialize(anno)

    def initialize(self, anno: np.ndarray):
        self.generate_specific_anno(anno)
        # self.generate_query_tree()
        # for i in range(len(self.voxel_array)):
        #     self.v_c_lut[i] = []
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
        self.voxel_array = np.argwhere(np.isin(anno, ch_id_list))

        return

    def filter_qnodes(self, q_nodes):
        '''
        filter nodes using bounding box of brain region
        :return:
        '''
        radius = __RADIUS__ / __RES__
        q_nodes = np.asarray(q_nodes)
        q_nodes = self._q_nodes_dim_check(q_nodes)
        xmin, ymin, zmin = np.min(self.voxel_array, axis=0) - radius
        xmax, ymax, zmax = np.max(self.voxel_array, axis=0) + radius
        # print((xmin, xmax), (ymin, ymax), (zmin, zmax))
        cond_mask = (q_nodes[:, 0] >= xmin) & (q_nodes[:, 1] >= ymin) & (q_nodes[:, 2] >= zmin) & \
                    (q_nodes[:, 0] <= xmax) & (q_nodes[:, 1] <= ymax) & (q_nodes[:, 2] <= zmax)

        new_q_nodes = deepcopy(q_nodes[cond_mask])

        return new_q_nodes

    def generate_query_tree(self, q_nodes, tree_type: str = 'balltree'):
        q_nodes = np.asarray(q_nodes)
        q_nodes = self._q_nodes_dim_check(q_nodes)
        qTree = None
        if tree_type.lower() == 'balltree':
            qTree = BallTree
        elif tree_type.lower() == 'kdtree':
            qTree = KDTree
        self.qtree = qTree(q_nodes[:, 0:3], metric='minkowski', p=2)

        return

    def generate_lut(self):
        radius = __RADIUS__ / __RES__

        self.v_c_lut = self.qtree.query_radius(self.voxel_array, r=radius)

        return

    # def generate_lut(self, q_nodes, ):
    #     radius = __RADIUS__ / __RES__
    #
    #     q_nodes = np.asarray(q_nodes)
    #     q_nodes = self._q_nodes_dim_check(q_nodes)
    #
    #     neighbors = self.qtree.query_radius(q_nodes[:, 0:3], r=radius)
    #
    #     return neighbors

    # def assign_lut(self, neighbors):
    #     voxel_compa_list = []
    #     for i in range(len(self.voxel_array)):
    #         voxel_compa_list.append([])
    #     for i in range(len(neighbors)):
    #         cur_voxel_list = neighbors[i]
    #         for voxel in cur_voxel_list:
    #             voxel_compa_list[voxel].append(voxel)
    #     self.v_c_lut = dict(zip(np.arange(0, len(self.voxel_array), 1),
    #                             voxel_compa_list))
    #     return


def generate_vcl(all_compartments, brain_region, anno) -> VoxelCompartmentLut:
    t0 = time.time()
    vcl = VoxelCompartmentLut(brain_region, anno)
    t1 = time.time()
    all_compartments = compa_preprocess(all_compartments, __RES__)
    t2 = time.time()
    filtered_compartments = vcl.filter_qnodes(all_compartments)
    print('filtered compartments array shape: ', filtered_compartments.shape)
    t3 = time.time()
    vcl.generate_query_tree(filtered_compartments)
    t4 = time.time()
    vcl.generate_lut()
    t5 = time.time()

    print(f'initialize vcl: {t1 - t0}\n',
          f'preprocess compartments: {t2 - t1}\n',
          f'filter compartments: {t3 - t2}\n',
          f'initialize query tree: {t4 - t3}\n',
          f'query (assign lut): {t5 - t4}',)
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
    cond = compa[:, 3] < 0
    compa[cond, 3:6] = -compa[cond, 3:6]

    if gpu:
        compa = compa.cpu().numpy()

    return compa


if __name__ == '__main__':
    import time

    anno = generate_anno(
        r'E:\ZhixiYun\Projects\GitHub\neuro_morpho_toolbox\neuro_morpho_toolbox\data\annotation_25.nrrd')
    all_compartments = np.load(r'G:\data\parcellation\binary\all_compartments.npy')
    print('all compartments array shape: ',all_compartments.shape)
    vcl = generate_vcl(all_compartments, 'CP', anno)
    print('brain region voxel size: ', len(vcl.voxel_array))
    print(vcl.v_c_lut)
