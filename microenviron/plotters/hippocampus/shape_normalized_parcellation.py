##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-16
#Description:               
##########################################################
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.sparse.csgraph import dijkstra

import cv2
import cc3d
import networkx as nx
from skan.csr import skeleton_to_csgraph
from skan import Skeleton, summarize
from tps import ThinPlateSpline
from scipy.spatial.transform import Rotation as R

from image_utils import get_mip_image
from file_io import load_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from anatomy.anatomy_vis import detect_edges3d
from image_utils import crop_nonzero_mask


def get_longest_skeleton(skel):
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
            con0 = (summ['node-id-src']==nid) | (summ['node-id-dst']==nid)
            con = con0 & (summ['branch-type']==1)
            con1 = con0 & (summ['branch-type'] != 1)
            if con.sum() <= 1: continue # process only

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
    p1 = summ_bak[['node-id-src', 'image-coord-src-0', 'image-coord-src-1', 'image-coord-src-2']].values
    p2 = summ_bak[['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1', 'image-coord-dst-2']].values
    pts_all = np.vstack((p1, p2))
    node1, node2 = summ[nid_keys].values[0]
    coords1 = pts_all[pts_all[:,0] == node1][0][1:]
    coords2 = pts_all[pts_all[:,0] == node2][0][1:]

    # get the path
    pgraph, coordinates = skeleton_to_csgraph(skel)
    coordinates = np.array(coordinates).transpose()
    id1 = np.nonzero((coordinates == coords1).sum(axis=1) == 3)[0][0]
    id2 = np.nonzero((coordinates == coords2).sum(axis=1) == 3)[0][0]
    # 
    dij = dijkstra(pgraph, directed=False, indices=[id1], return_predecessors=True)
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
    new_skel[pcoords[:,0], pcoords[:,1], pcoords[:,2]] = 1
    
    return new_skel, pcoords

def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    #SVD decomposition
    U, S, Vt = np.linalg.svd(translated_points)
    normal = Vt[-1]
    return centroid, normal

def get_rotated_anchors(pcoords):
    # estimate the rotation plane
    centroid, Vt = best_fit_plane(pcoords)
    # map all points to the plane
    projs = pcoords - np.dot(pcoords - centroid, Vt.reshape((3,1))) * Vt
    # calculate the angles to rotate
    vinp = projs[1:] - projs[0]
    vinp_n = vinp / np.linalg.norm(vinp, axis=1).reshape((-1, 1))
    cos_angs = vinp_n.dot(vinp_n[0])
    angs = np.arccos(cos_angs)
    mag = np.tan(-angs / 4)
    # get all rodrigue vectors
    rodrigues = mag.reshape((-1,1)) * Vt
    # rotate the original points
    rots = R.from_mrp(rodrigues)
    # relative coordinates
    prcoords = pcoords[1:] - pcoords[0]
    rcoords = rots.apply(prcoords)
    rcoords += pcoords[0]
    rcoords = np.vstack((pcoords[0], rcoords))
    return rcoords

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
    mask[zdim//2:] = 0
    # crop the mask for faster computing
    sub_mask, (zs,ze,ys,ye,xs,xe) = crop_nonzero_mask(mask, pad=1)
    
    print('===> get the skeleton')
    skel = skeletonize(sub_mask, method='lee')
    skel[skel > 0] = 1

    # iteratively prune the short leaves
    
    skel, pcoords = get_longest_skeleton(skel)

    if visualize:
        print('Saving images for visualization: ')
        orig_mip1 = get_mip_image(sub_mask, axis=0)
        orig_mip2 = get_mip_image(sub_mask, axis=1)
        orig_mip3 = get_mip_image(sub_mask, axis=2)

        mip1 = get_mip_image(skel, axis=0)
        mip2 = get_mip_image(skel, axis=1)
        mip3 = get_mip_image(skel, axis=2)

        m1 = np.hstack((orig_mip1, mip1)) * 255
        m2 = np.hstack((orig_mip2, mip2)) * 255
        m3 = np.hstack((orig_mip3, mip3)) * 255
        cv2.imwrite('mip1.png', m1)
        cv2.imwrite('mip2.png', m2)
        cv2.imwrite('mip3.png', m3)

    # stretch using thin-plate-spline
    rcoords = get_rotated_anchors(pcoords)

    init_dist = np.linalg.norm(pcoords[-1] - pcoords[0])
    final_dist = np.linalg.norm(rcoords[-1] - rcoords[0])
    import ipdb; ipdb.set_trace()
    print(f'Initial and Final distances between start-end points: {init_dist:.2f} and {final_dist:.2f}')
    
    # to avoid computation overwhelmming, use only the boundary points
    edges = detect_edges3d(sub_mask)
    ecoords = np.array(edges.nonzero()).transpose()
    tps = ThinPlateSpline(alpha=0.0)
    tps.fit(pcoords, rcoords)
    ecoords_t = tps.transform(ecoords)
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax.scatter(ecoords[:,0], ecoords[:,1], ecoords[:,2], alpha=0.5, color='blue')
        #ax.scatter(ecoords_t[:,0], ecoords_t[:,1], ecoords_t[:,2], alpha=0.5, color='orange')
        ax.scatter(rcoords[:,0], rcoords[:,1], rcoords[:,2], alpha=0.8, color='red')
        #ax.view_init(-60, -60)
        plt.savefig('temp.png')
        plt.close()
    
    

    return skel

    

if __name__ == '__main__':
    rname = 'CA1'
    #rname = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
    skeletonize_region(rname)


