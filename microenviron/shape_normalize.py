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
from skimage.measure import label

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



def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    #SVD decomposition
    U, S, Vt = np.linalg.svd(translated_points)
    normal = Vt[-1]
    return centroid, normal

def stretching3d(pcoords_o, ecoords_o, coords_o):
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

# the stretching function of both 3D/2D have some problems, I could not fix it at
# this moment. Try another way around.
def stretching2d(pcoords_o, ecoords_o, coords_o, visualize=False):
    '''
    pcoords: coordinates of medial axis points
    ecoords: coordinates of boundary points
    coords:  coordinates of neuronal points
    '''
    step = 10
    if pcoords_o.shape[0] < step:
        return pcoords_o, ecoords_o, coords_o
    pcoords = pcoords_o.copy()[::step]
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
    # find the a proper anchor point, along the main axis, but outside the current range
    pca = PCA()
    ecoords_t = pca.fit_transform(ecoords)
    maxid = ecoords_t[:,0].argmax()
    pa = 1.5 * (ecoords[maxid] - centroid) + centroid
    if visualize:
        plt.scatter([pa[0]], [pa[1]], c='red')
        plt.scatter([centroid[0]], [centroid[1]], c='orange')
        plt.scatter(ecoords_o[:,0], ecoords_o[:,1], c='green')
        plt.scatter(pcoords_o[:,0], pcoords_o[:,1], s=0.5, c='purple')

    # iteratively rotate to a line
    va = centroid - pa
    van = va / np.linalg.norm(va)
    output = []
    rbs = np.zeros_like(ecoords)
    for i in range(pcoords.shape[0]):
        pdi = pcoords[i]
        if i == 0:
            prev_cp = pa
        else:
            prev_cp = pcoords[i-1]
        vji = pcoords[i] - prev_cp
        # get the rotation matrix
        a = vji / np.linalg.norm(vji)
        b = van
        vv = np.cross(a,b)
        cc = np.dot(a, b)
        vx = np.array([cc, -vv, vv, cc]).reshape((2,2))
        
        prcoords = pcoords - prev_cp
        pcoords = vx.dot(prcoords.transpose()).transpose() + prev_cp

        output.append(pcoords[i])
        if i in top1_dict:
            ercoords = ecoords - prev_cp
            ecoords = vx.dot(ercoords.transpose()).transpose() + prev_cp
            plt.scatter(ecoords[:,0], ecoords[:,1], s=1, c='brown')
            rbs[top1_dict[i]] = ecoords[top1_dict[i]]

    output = np.array(output)

    # Do TPS interpolation
    np.random.seed(1024)
    random.seed(1024)
    # only using randomly selected 1000 points for fast interpolation
    ninter = 1000
    if ecoords_o.shape[0] > ninter:
        ids = np.arange(ecoords_o.shape[0])
        random.shuffle(ids)
        ids = ids[:ninter]
    else:
        ids = None

    tps = ThinPlateSpline(alpha=0.0)
    #tps.fit(ecoords_o[ids], rbs[ids])
    #rs = tps.transform(coords_o)
    rs = coords

    if visualize:
        plt.scatter(rbs[:,0], rbs[:,1])
        plt.savefig('temp.png')
        plt.close()

    return output, rbs, rs

def map_to_longitudinal_space(mask, pcoords, coords):
    # firstly, find the dorsal and ventral part
    mask = mask.astype(bool)
    skel = np.zeros_like(mask)
    skel[pcoords[:,0], pcoords[:,1]] = True
    separated_mask = mask & ~skel
    # Must use `connectivity=1`
    labeled_mask = label(separated_mask, connectivity=1)
    # in case of exceptions:
    if labeled_mask.max() > 2:
        # nearest neighbor interpolation
        exceptions = labeled_mask > 2
        ex_coords = np.array(np.nonzero(exceptions)).transpose()
        bases = (labeled_mask == 1) | (labeled_mask == 2)
        base_coords = np.array(np.nonzero(bases)).transpose()
        b_kdt = KDTree(base_coords, metric='euclidean')
        ex_m = b_kdt.query(ex_coords, k=1, return_distance=False)
        labeled_mask[exceptions.nonzero()] = labeled_mask[bases][ex_m.reshape(-1)]

    # make sure only two components exist
    assert(labeled_mask.max() == 2)
    part1 = labeled_mask == 1
    part2 = labeled_mask == 2
    # dorsal and ventral assignment based on their distance to the ccf atlas center
    c1 = np.array(np.nonzero(part1)).mean(axis=1)
    c2 = np.array(np.nonzero(part2)).mean(axis=1)
    
    # estimate their distance
    kdt = KDTree(pcoords, metric='euclidean')
    lcoords = np.hstack(kdt.query(coords, k=1, return_distance=True))
    vs = labeled_mask[coords[:,0], coords[:,1]]
    # for those outside the mask, we should interpolate their type
    fgm = vs > 0
    bgm = vs == 0
    bgc = coords[bgm]
    fgc = coords[fgm]
    # find the nearest fg point
    fg_kdt = KDTree(fgc, metric='euclidean')
    bgc_v = vs[fgm][fg_kdt.query(bgc, k=1, return_distance=True)[1]].reshape(-1)
    vs[bgm] = bgc_v

    img_center = np.array(mask.shape)/2
    d1 = np.linalg.norm(img_center - c1)
    d2 = np.linalg.norm(img_center - c2)
    if d1 > d2:
        lcoords[np.nonzero(vs==2)[0], 0] *= -1
    else:
        lcoords[np.nonzero(vs==1)[0], 0] *= -1
    
    return lcoords


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


