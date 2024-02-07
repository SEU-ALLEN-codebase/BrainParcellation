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

import leidenalg as lg
import igraph as ig



class BrainParcellation:
    def __init__(self, mefile, scale=25., full_features=True):
        self.df = self.load_features(mefile, scale=scale, full_features=full_features)


    def load_features(self, mefile, scale=25., full_features=True):
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

        return df

    def initialize_graph(self, load_partition=True):
        t0 = time.time()

        # Compute the sparse nearest neighbors graph
        # Adjust n_neighbors based on your dataset and memory constraints
        coords = self.df[['soma_x', 'soma_y', 'soma_z']]
        feats = self.df[self.fnames].to_numpy()
        # or try to use radius_neighbors_graph
        #A = kneighbors_graph(feats, n_neighbors=20, include_self=True, mode='distance', metric='euclidean', n_jobs=8)
        # the radius are in 25um space
        radius_th = 10.
        partition_file = 'community_memberships.json'

        A = radius_neighbors_graph(coords, radius=radius_th, include_self=True, mode='distance', metric='euclidean', n_jobs=8)
        print(f'[Neighbors generation]: {time.time() - t0:.2f} seconds')
        
        A_csr = csr_matrix(A)
        sources, targets = A_csr.nonzero()
        # estimate the edge weights
        dists = A_csr[sources, targets]
        dt = radius_th
        wd = np.squeeze(np.asarray(np.exp(-2.*dists/dt)))
        print('wd statis: ', wd.mean(), wd.max(), wd.min())
        print(f'Total and avg number of edges: {wd.shape[0]}, {wd.shape[0]/feats.shape[0]}')

        fs = feats[sources]
        ft = feats[targets]
        wf = np.exp(-5. * np.linalg.norm(fs - ft, axis=1))
        print('wf statis: ', wf.mean(), wf.max(), wf.min())
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
            partition = lg.RBConfigurationVertexPartition(g, membership=community_memberships)
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
        print(f'[Community statistics]: {community_sizes.mean():.1f}, {community_sizes.std():.1f}, {community_sizes.max()}, {community_sizes.min()}')
        comms, counts = np.unique(community_sizes, return_counts=True)
        print(comms, counts)

        # visualization
        #ig.plot(g, vertex_size=1, target='todel.png')
        #print(f'[Plot]: {time.time() - t0: .2f} seconds')

        node_to_community = {node: community for node, community in enumerate(partition.membership)}
        # Initialize a dictionary to hold lists of nodes for each community
        communities = defaultdict(list)

        # Populate the dictionary with node indices grouped by their community
        for node_index, community_index in enumerate(partition.membership):
            communities[community_index].append(node_index)

        
        


if __name__ == '__main__':
    mefile = './data/mefeatures_100K.csv'
    scale = 25.
    full_features = False
    
    bp = BrainParcellation(mefile, scale=scale, full_features=full_features)
    bp.initialize_graph()
    


