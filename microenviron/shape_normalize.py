##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-16
#Description:               
##########################################################
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA
from scipy.ndimage import convolve
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import KDTree
from skimage import morphology
from skimage.draw import line_nd

import cv2
import cc3d
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize
from scipy.spatial.transform import Rotation as R
from tps import ThinPlateSpline

from image_utils import get_mip_image
from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from anatomy.anatomy_vis import detect_edges3d, detect_edges2d
from image_utils import crop_nonzero_mask


def extend_skel_to_boundary(boundaries, pcoords, is_start=True):
    if is_start:
        pt = pcoords[0]
        pts_neighbor = pcoords[:10]
        vref = pts_neighbor[-1] - pts_neighbor[0]
    else:
        pt = pcoords[-1]
        pts_neighbor = pcoords[-10:]
        vref = pts_neighbor[0] - pts_neighbor[-1]

    # find the boundary point align well with the principal axis of skelenton
    pca = PCA()
    pca.fit(pts_neighbor)
    pc1 = pca.components_[0]
    if vref.dot(pc1) < 0:
        pc1 = -pc1
    
    # estimate the direction matchness
    vb = (pt - boundaries).astype(np.float64)
    vb /= (np.linalg.norm(vb, axis=1).reshape((-1,1)) + 1e-10)
    cos_dist = pc1.dot(vb.transpose())
    max_id = np.argmax(cos_dist)
    pta = boundaries[max_id]
    
    lpts = np.array(line_nd(pt, pta, endpoint=True)).transpose()[1:]
    if is_start:
        pcoords = np.vstack((lpts[::-1], pcoords))
    else:
        pcoords = np.vstack((pcoords, lpts))

    return pcoords

def get_longest_skeleton(mask, is_3D=True, extend_to_boundary=True, smoothing=True):
    mask = mask > 0 # only binary mask supported
    if smoothing:
        mask = morphology.closing(mask, morphology.square(5), mode='constant')

    # get the skeleton
    skel = skeletonize(mask, method='lee')
    skel[skel > 0] = 1
    
    # get the critical points: tip and multi-furcations
    summ = summarize(Skeleton(skel))
    summ_bak = summ.copy()
    summ = summ[['node-id-src', 'node-id-dst', 'branch-distance', 'branch-type']]
    nid_keys = ['node-id-src', 'node-id-dst']
    # iterative prune
    while summ.shape[0] >= 2:
        print(summ.shape[0])
        dcnts = dict(zip(*np.unique(summ[['node-id-src', 'node-id-dst']], return_counts = True)))
        for nid, cnt in dcnts.items():
            if cnt == 1: continue

            # remove possible circles
            ids, cnts = np.unique(summ[nid_keys].values, axis=0, return_counts=True)
            if (cnts > 1).sum() != 0:
                lcnts = cnts > 1
                for lids in ids[lcnts]:
                    dup_ones = (summ[nid_keys] == lids).sum(axis=1) == 2
                    nzi = np.nonzero(dup_ones)[0]
                    sub_summ = summ[dup_ones]
                    max_d_id = np.argmax(sub_summ['branch-distance'])
                    max_d_index = sub_summ.index[max_d_id]
                    to_drop = []
                    for idx in range(len(nzi)):
                        if idx != max_d_id:
                            to_drop.append(sub_summ.index[idx])
                    summ.drop(index=to_drop, inplace=True)
                    # check the type of current branch
                    nc_dict = dict(zip(*np.unique(summ[nid_keys], return_counts=True)))
                    if (nc_dict[lids[0]] != 1) and (nc_dict[lids[1]] != 1): 
                        summ.loc[max_d_index, 3] = 2
                    else:
                        summ.loc[max_d_index, 3] = 1
                    

            con0 = (summ['node-id-src']==nid) | (summ['node-id-dst']==nid)
            con = con0 & (summ['branch-type']==1)
            con1 = con0 & (summ['branch-type'] != 1)
            #if con.sum() <= 1: continue # process only

            if con1.sum() == 0:
                # keep the top two branches
                to_del = summ.index[np.argsort(summ['branch-distance'].values)[:-2]]
                summ.drop(index=to_del, inplace=True)
                # merge the last two segments
                nids, ncnts = np.unique(summ[nid_keys], return_counts=True)
                final_nids = nids[ncnts == 1]
                final_dist = summ['branch-distance'].sum()
                summ.drop(index=summ.index[0], inplace=True)
                summ.loc[summ.index, nid_keys] = final_nids
                summ.loc[summ.index, 'branch-distance'] = final_dist
            else:
                cur = summ[con]
                max_id = np.argmax(cur['branch-distance'])
                idx_max = cur.index[max_id]
                # remove several points
                to_del = [k for k in cur.index if k != idx_max]
                # remove items from dataframe
                summ.drop(index=to_del, inplace=True)
                # modify their features
                #import ipdb; ipdb.set_trace()
                if con0.sum() - con.sum() == 1:
                    # the current node is now a non-critical point, remove it
                    idx = con1[con1].index[0]
                    tt = summ.loc[idx]
                    tr = summ.loc[idx_max]
                    summ.loc[idx, 'branch-type'] = 1
                    summ.loc[idx, 'branch-distance'] = tt['branch-distance'] + tr['branch-distance']
                    stacks = np.hstack((tr[['node-id-src', 'node-id-dst']], tt[['node-id-src', 'node-id-dst']]))    
                    ids = [idx for idx in stacks if idx != nid]
                    summ.loc[idx, ['node-id-src', 'node-id-dst']] = ids
                    summ.drop(index=idx_max, inplace=True)
                elif con0.sum() - con.sum() == 0:
                    print('WARNING: ')
            

    # get the original information
    if is_3D:
        src_key = ['node-id-src', 'image-coord-src-0', 'image-coord-src-1', 'image-coord-src-2']
        dst_key = ['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1', 'image-coord-dst-2']
        vm = 3
    else:
        src_key = ['node-id-src', 'image-coord-src-0', 'image-coord-src-1']
        dst_key = ['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1']
        vm = 2

    # the two terminal points
    p1 = summ_bak[src_key].values
    p2 = summ_bak[dst_key].values
    pts_all = np.vstack((p1, p2))
    node1, node2 = summ[nid_keys].values[0]
    coords1 = pts_all[pts_all[:,0] == node1][0][1:]
    coords2 = pts_all[pts_all[:,0] == node2][0][1:]

    # get the path
    pgraph, coordinates = skeleton_to_csgraph(skel)
    coordinates = np.array(coordinates).transpose()
    id1 = np.nonzero((coordinates == coords1).sum(axis=1) == vm)[0][0]
    id2 = np.nonzero((coordinates == coords2).sum(axis=1) == vm)[0][0]
    # The skeletonization may result small circular points!
    dij = dijkstra(pgraph, directed=True, indices=[id1], return_predecessors=True)
    parents = dij[1][0]
    # transverse to the full path
    pids = []
    pid = id2
    while pid != -9999:
        pids.append(pid)
        pid = parents[pid]
    pcoords = coordinates[pids]    

    new_skel = skel.copy()
    new_skel.fill(0)
    if is_3D:
        new_skel[pcoords[:,0], pcoords[:,1], pcoords[:,2]] = 1
    else:
        new_skel[pcoords[:,0], pcoords[:,1]] = 1

    if extend_to_boundary:
        #---- extend the skeleton to boundary of image ----#
        if is_3D:
            edges = detect_edges3d(mask)
        else:
            edges = detect_edges2d(mask)
        ecoords = np.array(edges.nonzero()).transpose()

        pcoords = extend_skel_to_boundary(ecoords, pcoords, is_start=True)
        pcoords = extend_skel_to_boundary(ecoords, pcoords, is_start=False)
        # udate skeleton
        if is_3D:
            new_skel[pcoords[:,0], pcoords[:,1], pcoords[:,2]] = 1
        else:
            new_skel[pcoords[:,0], pcoords[:,1]] = 1       
    
    return new_skel, pcoords

def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    #SVD decomposition
    U, S, Vt = np.linalg.svd(translated_points)
    normal = Vt[-1]
    return centroid, normal

def stretching(pcoords_o, ecoords_o, coords_o):
    '''
    pcoords: coordinates of medial axis points
    ecoords: coordinates of boundary points
    coords:  coordinates of neuronal points
    '''
    pcoords = pcoords_o.copy()
    ecoords = ecoords_o.copy()
    coords = coords_o.copy()
    # get the correspondence between points and ecoords
    kdt = KDTree(pcoords, metric='euclidean')
    top1 = kdt.query(ecoords, k=1, return_distance=False)
    top1_dict = {}
    for vt in np.unique(top1):
        top1_dict[vt] = np.nonzero(top1 == vt)[0]

    # estimate the rotation plane
    centroid, Vt = best_fit_plane(pcoords)
    # map all points to the plane
    projs = pcoords - np.dot(pcoords - centroid, Vt.reshape((3,1))) * Vt
    # find the a proper anchor point, along the main axis, but outside the current range
    pca = PCA()
    pca.fit(projs)
    projs_x = np.dot(projs, pca.components_[0])
    maxid = projs_x.argmax()
    pa = 1.2 * (projs[maxid] - centroid) + centroid

    # iteratively rotate to a line
    va = centroid - pa
    van = va / np.linalg.norm(va)
    output = []
    rbs = np.zeros_like(ecoords)
    for i in range(pcoords.shape[0]):
        pji = projs[i]
        pdi = pcoords[i]
        if i == 0:
            prev_cj = pa
            prev_cp = pa
        else:
            prev_cj = projs[i-1]
            prev_cp = pcoords[i-1]
        vji = projs[i] - prev_cj
        # get the rotation matrix
        a = vji / np.linalg.norm(vji)
        b = van
        v = np.cross(a,b)
        c = np.dot(a, b)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rmat = np.eye(3) + vx + np.dot(vx, vx) * ((1-c)/np.linalg.norm(v)**2)
        rots = R.from_matrix(rmat)
        
        prcoords = pcoords - prev_cp
        prrojs = projs - prev_cj
        pcoords = rots.apply(prcoords) + prev_cp
        projs = rots.apply(prrojs) + prev_cj

        output.append(pcoords[i])
        if i in top1_dict:
            ercoords = ecoords - prev_cp
            ecoords = rots.apply(ercoords) + prev_cp
            rbs[top1_dict[i]] = ecoords[top1_dict[i]]

    # Do TPS interpolation
    np.random.seed(1024)
    random.seed(1024)
    # only using randomly selected 1000 points for fast interpolation
    ninter = 1000
    if ecoords_o.shape[0] > ninter:
        ids = np.arange(ecoords_o.shape[0])
        random.shuffle(ids)
        ids = ids[:ninter]

    tps = ThinPlateSpline(alpha=0.0)
    tps.fit(ecoords_o[ids], rbs[ids])
    rs = tps.transform(coords_o)

    return np.array(output), rbs,rs

def get_skeletons(reg_mask, visualize=False):
    print('===> get the skeleton')
    skel = skeletonize(reg_mask, method='lee')
    skel[skel > 0] = 1

    skel, pcoords = get_longest_skeleton(skel)

    if visualize:
        print('Saving images for visualization: ')
        orig_mip1 = get_mip_image(reg_mask, axis=0)
        orig_mip2 = get_mip_image(reg_mask, axis=1)
        orig_mip3 = get_mip_image(reg_mask, axis=2)

        mip1 = get_mip_image(skel, axis=0)
        mip2 = get_mip_image(skel, axis=1)
        mip3 = get_mip_image(skel, axis=2)

        m1 = np.hstack((orig_mip1, mip1)) * 255
        m2 = np.hstack((orig_mip2, mip2)) * 255
        m3 = np.hstack((orig_mip3, mip3)) * 255
        cv2.imwrite('mip1.png', m1)
        cv2.imwrite('mip2.png', m2)
        cv2.imwrite('mip3.png', m3)

    
    # to avoid computation overwhelmming, use only the boundary points
    edges = detect_edges3d(reg_mask)
    ecoords = np.array(edges.nonzero()).transpose()

    return pcoords, ecoords

def shape_normalized_scaling_bak(reg_mask, coords=None, visualize=False):
    pcoords, ecoords = get_skeletons(reg_mask, visualize=visualize)

    if coords is None:
        coords = ecoords
    rcoords, ecoords_t, coords_t = stretching(pcoords, ecoords, coords)
    # scaling
    pca = PCA()
    ecoords_t = pca.fit_transform(ecoords_t)
    coords_t = pca.transform(coords_t)
    stds = np.sqrt(pca.explained_variance_)
    scales = stds.sum() / stds
    print(f'Scales are: {scales}')
    # scaling 
    coords_t = scales * coords_t
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(ecoords_t[:,0], ecoords_t[:,1], ecoords_t[:,2], alpha=0.5, color='blue')
        #ax.scatter(coords_t[:,0], coords_t[:,1], coords_t[:,2], alpha=0.5, color='orange')
        #ax.scatter(rcoords[:,0], rcoords[:,1], rcoords[:,2], alpha=0.8, color='red')
        ax.view_init(-60, -60)
        plt.savefig('temp.png')
        plt.close()

    return coords_t

def shape_normalized_scaling(reg_mask, coords=None, visualize=False):
    edges = detect_edges3d(reg_mask)
    ecoords = np.array(edges.nonzero()).transpose()
    
    pca = PCA()
    pca.fit(ecoords)
    if coords is None:
        coords = ecoords
    coords_t = pca.transform(coords)
    stds = np.sqrt(pca.explained_variance_)
    scales = stds.sum() / stds
    print(f'Scales are: {scales}')
    # scaling
    coords_t = scales * coords_t
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(coords_t[:,0], coords_t[:,1], coords_t[:,2], alpha=0.5, color='orange')
        #ax.view_init(-60, -60)
        plt.savefig('temp.png')
        plt.close()

    return coords_t


def skeletonize_region(rname, visualize=True):
    """
    @args mask: binary mask to get the skeleton
    """
    print('--> Loading the atlas and anatomical information')
    atlas = load_image(MASK_CCF25_FILE)
    ana_tree = parse_ana_tree(keyname='name')
    
    print('--> Get the mask')
    if type(rname) is list:
        mask = np.zeros_like(atlas).astype(bool)
        for rn in rname:
            rid = ana_tree[rn]['id']
            mask = mask | (atlas == rid)
    else:
        mask = atlas == ana_tree[rname]['id']
    
    # keep only right hemisphere
    zdim = mask.shape[0]
    mask[:zdim//2] = 0
    # crop the mask for faster computing
    sub_mask, (zs,ze,ys,ye,xs,xe) = crop_nonzero_mask(mask, pad=0)
    
    coords_t = shape_normalized_scaling(sub_mask, visualize=visualize)

    #init_dist = np.linalg.norm(pcoords[-1] - pcoords[0])
    #final_dist = np.linalg.norm(rcoords[-1] - rcoords[0])
    #print(f'Initial and Final distances between start-end points: {init_dist:.2f} and {final_dist:.2f}')
    
    return 

    

if __name__ == '__main__':
    rname = 'CA1'
    #rname = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
    skeletonize_region(rname)


