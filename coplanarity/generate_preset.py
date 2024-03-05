import glob
import os
import pandas as pd
import numpy as np
from utils import Neuron


def generate_compartment(neuron: Neuron, ) -> np.ndarray:
    index_accumulate_path_dist = np.zeros(len(neuron), dtype=np.float32) - 1
    index_accumulate_path_dist_center = np.zeros(len(neuron), dtype=np.float32) - 1
    index_compartment_order = np.zeros(len(neuron), dtype=np.float32)
    index_distance = np.zeros(len(neuron), dtype=np.float32) - 1
    index_pos = np.zeros((len(neuron), 3), dtype=np.float32)
    index_vector_norm = index_pos.copy()

    root_node = neuron.soma
    root_node_id = root_node[0]
    root_node_idx = neuron.nidHash[root_node_id]
    root_ch_list = neuron.indexChildren[root_node_idx]

    index_accumulate_path_dist[root_node_idx] = 0
    index_compartment_order[root_node_idx] = 0

    ch_idx_stack = list(root_ch_list)

    excluded_list = [root_node_idx]

    cur_node = root_node
    cur_node_id = root_node_id
    cur_node_idx = root_node_idx
    cur_ch_list = root_ch_list

    def _calc_v_and_dist(node1, node2, ):
        # distance (same as tmp_path_dist) between two nodes and their unit direction vector
        vector = node2[2:5] - node1[2:5]
        tmp_path_dist = np.linalg.norm(vector).astype(np.float32)
        if tmp_path_dist == 0:
            vector_norm = vector
        else:
            vector_norm = (vector / tmp_path_dist).astype(np.float32)

        return tmp_path_dist, vector_norm

    while ch_idx_stack:

        next_node_idx = ch_idx_stack.pop()
        next_node = neuron.swc[next_node_idx]
        next_node_id = next_node[0]
        next_ch_list = neuron.indexChildren[next_node_idx]

        next_node_pid = next_node[6]
        next_node_pid_idx = neuron.nidHash[next_node_pid]
        next_node_pid_node = neuron.swc[next_node_pid_idx]

        # update attributes
        tmp_path_dist, vector_norm = _calc_v_and_dist(next_node_pid_node, next_node)
        if tmp_path_dist == 0:
            excluded_list.append(next_node_idx)
        index_accumulate_path_dist[next_node_idx] = index_accumulate_path_dist[next_node_pid_idx] + tmp_path_dist
        index_accumulate_path_dist_center[next_node_idx] = index_accumulate_path_dist[
                                                               next_node_pid_idx] + tmp_path_dist / 2
        index_compartment_order[next_node_idx] = index_compartment_order[next_node_pid_idx] + 1

        index_pos[next_node_idx] = next_node_pid_node[2:5]
        index_vector_norm[next_node_idx] = vector_norm
        index_distance[next_node_idx] = tmp_path_dist

        cur_node = next_node
        cur_node_id = next_node_id
        cur_node_idx = next_node_idx
        cur_ch_list = next_ch_list

        while len(cur_ch_list) == 1:
            next_node_idx = cur_ch_list[0]
            next_node = neuron.swc[next_node_idx]
            next_node_id = next_node[0]
            next_ch_list = neuron.indexChildren[next_node_idx]

            # update attributes
            tmp_path_dist, vector_norm = _calc_v_and_dist(cur_node, next_node)
            if tmp_path_dist == 0:
                excluded_list.append(next_node_idx)
            index_accumulate_path_dist[next_node_idx] = index_accumulate_path_dist[cur_node_idx] + tmp_path_dist
            index_accumulate_path_dist_center[next_node_idx] = index_accumulate_path_dist[
                                                                   cur_node_idx] + tmp_path_dist / 2
            index_compartment_order[next_node_idx] = index_compartment_order[cur_node_idx]
            index_pos[next_node_idx] = cur_node[2:5]
            index_vector_norm[next_node_idx] = vector_norm
            index_distance[next_node_idx] = tmp_path_dist

            cur_node = next_node
            cur_node_id = next_node_id
            cur_node_idx = next_node_idx
            cur_ch_list = next_ch_list

        # tip node or bifurcation (furcation) node
        if len(cur_ch_list) == 0 or len(cur_ch_list) >= 2:
            if len(cur_ch_list) == 0:
                continue  # pass
            else:
                ch_idx_stack.extend(cur_ch_list)

    without_soma_condition = ~np.isin(np.arange(0, len(neuron), 1), excluded_list)
    compartment = np.concatenate([index_pos[without_soma_condition],
                                  index_vector_norm[without_soma_condition],
                                  index_distance[without_soma_condition][:, np.newaxis],
                                  index_accumulate_path_dist_center[without_soma_condition][:, np.newaxis],
                                  index_compartment_order[without_soma_condition][:, np.newaxis]], axis=1,
                                 dtype=np.float32)

    return compartment


if __name__ == '__main__':
    # dump_swc_to_npy(r'G:\data\parcellation\cropped_100um', r'G:\data\parcellation\pickle\all_swc.pickle')

    compartments = []
    folder_path = r'G:\data\parcellation\cropped_100um_filtered'
    fplist = glob.glob(os.path.join(folder_path, '*', '*.swc'))
    fplist.sort()
    for fp_swc in fplist:
        # print(fp_swc)
        neuron = Neuron(fp_swc, scale=1, mirror=True)
        compartment = generate_compartment(neuron)
        compartments.extend(compartment.tolist())
    compartments = np.array(compartments, dtype=np.float32)
    np.save(r'G:\data\parcellation\binary\all_compartments.npy', compartments, )

if __name__ == '_1_main11__':

    df = pd.read_csv('../evaluation/data/final_filtered_swc.txt').values.flatten()
    import shutil

    tar_path = r'G:\data\parcellation\cropped_100um_filtered'
    src_path = r'G:\data\parcellation\cropped_100um'
    fplist = glob.glob(os.path.join(src_path, '*', '*.swc'))
    fplist.sort()
    for fp in fplist:
        fn = os.path.split(fp)[-1]
        fn = fn.replace('_stps.swc', '')
        if fn in df:
            tp = fp.replace('cropped_100um', 'cropped_100um_filtered')
            tar_folder = os.path.split(tp)[0]
            if not os.path.exists(tar_folder):
                os.makedirs(tar_folder)
            shutil.copy(fp, tp)
