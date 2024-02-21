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
import matplotlib as mpl
import matplotlib.cm as cm
import cv2

import leidenalg as lg
import igraph as ig

from image_utils import get_mip_image, image_histeq
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_vis import detect_edges2d

from generate_me_map import process_mip


class BrainParcellation:
    def __init__(self, mefile, scale=25., full_features=True, flipLR=True, seed=1024):
        self.df = self.load_features(mefile, scale=scale, full_features=full_features, flipLR=flipLR)
        np.random.seed(seed)


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


    def community_detection(self, load_partition=True):
        t0 = time.time()

        # Compute the sparse nearest neighbors graph
        # Adjust n_neighbors based on your dataset and memory constraints
        coords = self.df[['soma_x', 'soma_y', 'soma_z']]
        feats = self.df[self.fnames].to_numpy()

        partition_file = 'community_memberships.json'
        
        feature_filter = False
        if feature_filter:
            ngb_dist = 4
            ngbs = radius_neighbors_graph(coords.values, radius=ngb_dist, include_self=True, 
                        mode='distance', metric='euclidean', n_jobs=8)
            ngbs = csr_matrix(ngbs)
           
            # do filtering
            sources, targets = ngbs.nonzero()
            stdict = {}
            for s, t in zip(sources, targets):
                if s in stdict:
                    stdict[s].append(t)
                else:
                    stdict[s] = [t]
            # feature level median filtering
            new_feats = feats.copy()
            for s, ts in stdict.items():
                if len(ts) > 1:
                    ts = [s] + ts
                    cur_fs = feats[ts]
                    mf = np.median(cur_fs, axis=0)
                    # select the one most similar to median feature vector
                    #import ipdb; ipdb.set_trace()
                    dist2mf = np.linalg.norm(cur_fs - mf, axis=1)
                    new_mf = cur_fs[np.argmin(dist2mf)]
                
                    new_feats[s] = new_mf
                
            feats = new_feats
            print(f'Average number of neighbors: {len(sources)/coords.shape[0]:.2f}')
            print(f'[Feature filtering]: {time.time() - t0:.2f} seconds')

        
        # or try to use radius_neighbors_graph
        # the radius are in 25um space
        radius_th = 10.
        par1 = 3.
        par2 = 5.

        #A = kneighbors_graph(coords, n_neighbors=50, include_self=True, mode='distance', metric='euclidean', n_jobs=8)
        A = radius_neighbors_graph(coords.values, radius=radius_th, include_self=True, mode='distance', metric='euclidean', n_jobs=8)
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
        
        # partition object does not support serialization with json, so we use json to walk around
        if load_partition and os.path.exists(partition_file):
            print('>>> Loading partition')
            with open(partition_file, 'r') as fp:
                community_memberships = json.load(fp)
            # Create a new partition object with the loaded memberships
            partition = lg.ModularityVertexPartition(g, 
                                initial_membership=community_memberships, 
                                weights=weights)
        else:
            ### Step 3: Apply the Leiden Algorithm
            partition = lg.find_partition(g, lg.ModularityVertexPartition, weights='weight')
            print(f'[Partition]: {time.time() - t0: .2f} seconds')


            print('>>> Saving partition')
            community_memberships = partition.membership

            # Save community memberships using JSON
            with open(partition_file, 'w') as f:
                json.dump(community_memberships, f)

        community_sizes = np.array([len(community) for community in partition])
        print(f'[Number of communities] = {len(partition)}')
        print(f'[Number of large communities (n>5)] = {(community_sizes > 5).sum()}')
        print(f'[Community statistics: mean/std/max/min]: {community_sizes.mean():.1f}, {community_sizes.std():.1f}, {community_sizes.max()}, {community_sizes.min()}')
        comms, counts = np.unique(community_sizes, return_counts=True)
        print(comms, counts)

        # visualization the graph
        #ig.plot(g, vertex_size=1, target='todel.png')
        #print(f'[Plot]: {time.time() - t0: .2f} seconds')

        node_to_community = {node: community for node, community in enumerate(partition.membership)}
        # Initialize a dictionary to hold lists of nodes for each community
        communities = defaultdict(list) # community to node

        # Populate the dictionary with node indices grouped by their community
        for node_index, community_index in enumerate(partition.membership):
            communities[community_index].append(node_index)

        # plot onto the CCF space
        dfp = coords.copy()
        mask = load_image(MASK_CCF25_FILE)  # z,y,x order!
        dfp['parc'] = partition.membership
        self.parcellate(dfp, mask)
        self.visualize_on_ccf(dfp, mask)
        print()
        
    def parcellate(self, dfp, mask):
        zdim, ydim, xdim = mask.shape
        zdim2, ydim2, xdim2 = zdim // 2, ydim // 2, xdim // 2
        lmask = mask > 0; lmask[:zdim2] = 0
        lindices = np.where(lmask)
        interp = NearestNDInterpolator(dfp[['soma_z', 'soma_y', 'soma_x']], dfp['parc'])
        predv = interp(*lindices)
        
        lmask = lmask.astype(np.uint16)
        lmask[lindices] = predv

        # colorizing
        lnz = lmask.nonzero()
        crds = np.stack(lnz).transpose()
        values = lmask[lnz]
        cmask = self.random_colorize(crds, values, mask.shape, values.max())
        save_image('parc3d.tif', cmask, useCompression=True)

        # visualize
        for i, dim in zip(range(3), (zdim2, ydim2, xdim2)):
            m2d0 = np.take(cmask, dim, i)
            # overlay the boundaries
            m2d1 = np.take(mask, dim, i)
            edges = detect_edges2d(m2d1)
            p1 = m2d0.copy()
            p1[edges] = np.array([0,0,0,255])
            outfile = f'parc_axid{i}.png'
            cv2.imwrite(outfile, p1)
            if i != 0:
                print(f'Rotate by 90 degree')
                os.system(f'convert {outfile} -rotate 90 {outfile}')

        print()


if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    scale = 25.
    full_features = False
    load_partition = False
    
    bp = BrainParcellation(mefile, scale=scale, full_features=full_features)
    bp.community_detection(load_partition=load_partition)
    


