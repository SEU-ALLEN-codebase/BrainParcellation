##########################################################
#Author:          Yufeng Liu
#Create time:     2024-02-04
#Description:               
##########################################################

import os
import time
import json
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import Voronoi, distance_matrix
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import cc3d
from skimage.morphology import ball as morphology_ball
from skimage.morphology import cube as morphology_cube
from skimage.filters.rank import median as rank_median_filter

import leidenalg as lg
import igraph as ig

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_vis import detect_edges2d
from math_utils import min_distances_between_two_sets

from generate_me_map import process_mip


class BrainParcellation:
    def __init__(self, mefile, scale=25., full_features=True, flipLR=True, seed=1024):
        self.df = self.load_features(mefile, scale=scale, full_features=full_features, flipLR=flipLR)
        self.seed = seed
        np.random.seed(seed)
        self.flipLR = flipLR


    def load_features(self, mefile, scale=25., full_features=True, flipLR=True):
        df = pd.read_csv(mefile, index_col=0)
        
        if full_features:
            cols = df.columns
            self.fnames = [fname for fname in cols if fname[-3:] == '_me']
        else:
            # Features selected by mRMR
            self.fnames = ['Length_me', 'AverageFragmentation_me', 'AverageContraction_me']
    
        # standardize
        tmp = df[self.fnames]
        tmp = (tmp - tmp.mean()) / (tmp.std() + 1e-10)
        df[self.fnames] = tmp

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

        return df

    @staticmethod
    def random_colorize(coords, values, shape3d, color_level):
        # map the communities to different colors using randomized color map
        norm = mpl.colors.Normalize(values.min(), vmax=values.max())
        vals = np.linspace(0,1,color_level)
        np.random.shuffle(vals)
        cmap = cm.colors.ListedColormap(cm.jet(vals))
        #cmap = cm.bwr
        smapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = np.floor(smapper.to_rgba(values) * 255).astype(np.uint8)
        zyx = np.floor(coords).astype(np.int32)

        # intialize map
        pmap = np.zeros((*shape3d, colors.shape[-1]), dtype=np.uint8)
        pmap[zyx[:,0], zyx[:,1], zyx[:,2]] = colors
    
        return pmap

    def visualize_on_ccf(self, dfp, mask):
        shape3d = mask.shape
        zdim2, ydim2, xdim2 = shape3d[0]//2, shape3d[1]//2, shape3d[2]//2

        crds = dfp[['soma_z', 'soma_y', 'soma_x']]
        values = dfp['parc']
        pmap = self.random_colorize(crds.to_numpy(), values, shape3d, 5120)
        
        thickX2 = 20
        for axid in range(3):
            print(f'--> Processing axis: {axid}')
            cur_map = pmap.copy()
            if thickX2 != -1:
                if axid == 0:
                    cur_map[:zdim2-thickX2] = 0
                    cur_map[zdim2+thickX2:] = 0
                elif axid == 1:
                    cur_map[:,:ydim2-thickX2] = 0
                    cur_map[:,ydim2+thickX2:] = 0
                else:
                    cur_map[:,:,:xdim2-thickX2] = 0
                    cur_map[:,:,xdim2+thickX2:] = 0
            print(cur_map.mean(), cur_map.std())

            mip = get_mip_image(cur_map, axid)
            figname = f'temp_mip{axid}.png'
            process_mip(mip, mask, axis=axid, figname=figname, mode='composite')
            # load and remove the zero-alpha block
            img = cv2.imread(figname, cv2.IMREAD_UNCHANGED)
            wnz = np.nonzero(img[img.shape[0]//2,:,-1])[0]
            ws, we = wnz[0], wnz[-1]
            hnz = np.nonzero(img[:,img.shape[1]//2,-1])[0]
            hs, he = hnz[0], hnz[-1]
            img = img[hs:he+1, ws:we+1]
            if axid != 0:   # rotate 90
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # set the alpha of non-brain region as 0
            img[img[:,:,-1] == 1] = 0
            cv2.imwrite(figname, img)

    @staticmethod
    def remove_small_parcs(sub_mask, sub_fg_mask, cc_mask=None, cc_ids=None, cc_cnts=None, min_pts=0):
        if cc_mask is None:
            cc_mask = cc3d.connected_components(sub_mask, connectivity=26)
            # we would like to re-estimate the small communities/components
            cc_ids, cc_cnts = np.unique(cc_mask, return_counts=True)
        
        have_small_parcs = False
        for cc_id, cc_cnt in zip(cc_ids, cc_cnts):
            if cc_cnt < min_pts:
                try:
                    interp_mask = interp_mask | (cc_mask == cc_id)
                except UnboundLocalError:
                    interp_mask = cc_mask == cc_id
                have_small_parcs = True

        if have_small_parcs:
            anchor_mask = sub_fg_mask ^ interp_mask
            anchor_coords = np.array(np.where(anchor_mask)).transpose()
            anchor_values = sub_mask[anchor_mask]
            interp = NearestNDInterpolator(anchor_coords, anchor_values)
            predv = interp(*np.where(interp_mask))
            sub_mask[interp_mask] = predv

            # re-estimation
            cc_mask = cc3d.connected_components(sub_mask, connectivity=26)
            # we would like to re-estimate the small communities/components
            cc_ids, cc_cnts = np.unique(cc_mask, return_counts=True)

        return sub_mask, cc_mask, cc_ids, cc_cnts
        
    def save_colorized_images(self, cmask, mask):
        zdim, ydim, xdim = mask.shape
        zdim2, ydim2, xdim2 = zdim // 2, ydim // 2, xdim // 2
        # visualize
        for i, dim in zip(range(3), (zdim2, ydim2, xdim2)):
            for j in range(3):
                k = dim + (j-1)*40
                m2d0 = np.take(cmask, k, i)
                # overlay the boundaries
                m2d1 = np.take(mask, k, i)
                edges = detect_edges2d(m2d1)
                p1 = m2d0.copy()
                p1[edges] = np.array([0,0,0,255])
                outfile = f'parc_axid{i}_{j}.png'
                cv2.imwrite(outfile, p1)
                if i != 0:
                    print(f'Rotate by 90 degree')
                    os.system(f'convert {outfile} -rotate 90 {outfile}')

        print()



    def community_detection(self, load_partition=True):
        t0 = time.time()
        mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
        lmask = mask.copy() # left mask
        mshape = mask.shape
        lmask[:mshape[0]//2] = 0

        # Compute the sparse nearest neighbors graph
        # Adjust n_neighbors based on your dataset and memory constraints
        coords_all = self.df[['soma_x', 'soma_y', 'soma_z']]
        feats_all = self.df[self.fnames].to_numpy()

        # using CP to debug
        regid = 672 # CP
        cp_mask = self.df['region_id_r316'] == regid
        coords = coords_all[cp_mask]
        feats = feats_all[cp_mask]

        # or try to use radius_neighbors_graph
        # the radius are in 25um space
        radius_th = 10.
        par1 = 3.
        par2 = 3.
        min_pts_per_parc = 4**3 # 10^6 um^3

        #A = radius_neighbors_graph(coords.values, radius=radius_th, include_self=True, mode='distance', metric='euclidean', n_jobs=8)
        A = kneighbors_graph(coords.values, n_neighbors=80, include_self=True, mode='distance', metric='euclidean', n_jobs=8)
        print(f'[Neighbors generation]: {time.time() - t0:.2f} seconds')
        
        A_csr = csr_matrix(A)
        sources, targets = A_csr.nonzero()
        # estimate the edge weights
        dists = A_csr[sources, targets]
        dt = radius_th
        wd = np.squeeze(np.asarray(np.exp(-par1*dists/dt)))
        print(f'wd[mean/max/min]: {wd.mean():.2f}, {wd.max():.2f}, {wd.min():.2f}')
        print(f'Total and avg number of edges: {wd.shape[0]}, {wd.shape[0]/feats.shape[0]:.2f}')

        fs = feats[sources]
        ft = feats[targets]
        wf = np.exp(-par2 * np.linalg.norm(fs - ft, axis=1))
        print(f'wf[mean/max/min]: {wf.mean():.2f}, {wf.max():.2f}, {wf.min():.2g}')
        print(f'[weights estimation]: {time.time() - t0:.2f} seconds')

        weights = wd * wf
        

        g = ig.Graph(list(zip(sources, targets)), directed=False)
        g.es['weight'] = weights
        print(f'[Graph initialization]: {time.time() - t0: .2f} seconds')
        
        ### Step 3: Apply the Leiden Algorithm
        partition = lg.find_partition(g, lg.ModularityVertexPartition, weights='weight', seed=self.seed)
        print(f'[Partition]: {time.time() - t0: .2f} seconds')


        print('>>> Saving partition')
        community_memberships = [i+1 for i in partition.membership] # re-indexing starting from 1
        min_pts_per_comm = np.sqrt(coords.shape[0])

        community_sizes = np.array([len(community) for community in partition])
        print(f'[Number of communities] = {len(partition)}')
        print(f'[Number of large communities (n>{min_pts_per_comm:.2f})] = {(community_sizes > min_pts_per_comm).sum()}')
        print(f'[Community statistics: mean/std/max/min]: {community_sizes.mean():.1f}, {community_sizes.std():.1f}, {community_sizes.max()}, {community_sizes.min()}')
        comms, counts = np.unique(community_sizes, return_counts=True)
        print(comms, counts)

        node_to_community = {node: community for node, community in enumerate(community_memberships)}
        # Initialize a dictionary to hold lists of nodes for each community
        communities = defaultdict(list) # community to node

        # Populate the dictionary with node indices grouped by their community
        for node_index, community_index in enumerate(community_memberships):
            communities[community_index].append(node_index)

        # estimate the weighted center of each community
        mcoords = []
        mnodes = []
        mcomms = []
        for icomm, inodes in communities.items():
            if len(inodes) < min_pts_per_comm:
                continue
            cur_coords = coords.iloc[inodes]
            mnodes.extend(inodes)
            mcomms.extend([icomm]*len(inodes))
            mcoord = cur_coords.mean(axis=0).values
            mcoords.append(mcoord)
        mcoords = np.array(mcoords)

        # assign the current region into Voronoi cells
        if self.flipLR:
            reg_mask = (lmask == regid)
        else:
            reg_mask = (mask == regid)
        nzcoords = reg_mask.nonzero()
        nzcoords_t = np.array(nzcoords).transpose()

        parc_method = 'NearestNeighbor'
        if parc_method == 'Voronoi':
            dms, dmi = min_distances_between_two_sets(nzcoords_t, mcoords, topk=1, reciprocal=False, return_index=True, tree_type='BallTree')
            cmask = self.random_colorize(nzcoords_t, dmi[:,0], reg_mask.shape, dmi.max())
        elif parc_method == 'NearestNeighbor':
            #interp = NearestNDInterpolator(coords[['soma_z', 'soma_y', 'soma_x']], partition.membership)
            interp = NearestNDInterpolator(coords.iloc[mnodes][['soma_z', 'soma_y', 'soma_x']].values, mcomms)
            predv = interp(*nzcoords)
            print(f'[Interpolation]: {time.time() - t0:.2f} seconds')

            cur_mask = reg_mask.astype(np.uint8)
            cur_mask[nzcoords] = predv
            #import ipdb; ipdb.set_trace()
            # We should only processing the mask region, so pre-extracting it.
            zmin, ymin, xmin = nzcoords_t.min(axis=0)
            zmax, ymax, xmax = nzcoords_t.max(axis=0)
            #import ipdb; ipdb.set_trace()
            sub_mask = cur_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
            print(f'[Initial No. of CCs]: {len(np.unique(cc3d.connected_components(sub_mask, connectivity=26)))-1}')

            # median filtering
            # Python version on CPU. GPU implementation using PyTorch should be much faster, 
            # we can refer to https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598 
            # for more specific implementation.
            sub_fg_mask = sub_mask > 0

            # 2 rounds of parcellation smoothing
            cc_mask, cc_ids, cc_cnts = None, None, None
            for i_interp in range(2):
                sub_mask = rank_median_filter(sub_mask, morphology_ball(5), mask=sub_fg_mask)
                sub_mask[~sub_fg_mask] = 0  # force to zero

                sub_mask, cc_mask, cc_ids, cc_cnts = self.remove_small_parcs(sub_mask, sub_fg_mask, cc_mask, cc_ids, cc_cnts, min_pts_per_parc)
                print(f'[No. of CCs for iter={i_interp}]: {cc_mask.max()}')
                print(cc_ids, cc_cnts)

            # further remove small parcellations
            while (cc_cnts<min_pts_per_parc).sum() > 0:
                sub_mask, cc_mask, cc_ids, cc_cnts = self.remove_small_parcs(sub_mask, sub_fg_mask, cc_mask, cc_ids, cc_cnts, min_pts_per_parc)
            
            print(f'[No. of CCs finally]: {cc_mask.max()}')
            print(cc_ids, cc_cnts)

            # re-ordering the connected components or parcellations
            fg_ids = np.unique(cc_mask[sub_fg_mask])
            for i, fg_id in enumerate(fg_ids):
                sub_mask[cc_mask == fg_id] = i+1
            sub_mask[~sub_fg_mask] = 0

            cur_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = sub_mask
            # 
            cmask = self.random_colorize(nzcoords_t, cur_mask[nzcoords], mask.shape, predv.max())
            print(f'Communities after filtering: {len(np.unique(sub_mask))-1}')

        self.save_colorized_images(cmask, mask)
        print(f'[After colorization]: {time.time() - t0:.2f} seconds')

    
    
if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    scale = 25.
    full_features = False
    load_partition = False
    
    bp = BrainParcellation(mefile, scale=scale, full_features=full_features)
    bp.community_detection(load_partition=load_partition)
    


