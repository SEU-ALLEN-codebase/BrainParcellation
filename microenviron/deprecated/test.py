##########################################################
#Author:          Yufeng Liu
#Create time:     2024-02-04
#Description:               
##########################################################
import igraph as ig
import leidenalg as la
import time
import numpy as np

from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree

'''
for nnodes in [500, 1000, 5000]:
    for p2 in [0.001, 0.01, 0.1]:
        G = ig.Graph.Erdos_Renyi(nnodes, p2)
        t0 = time.time()
        partition = la.find_partition(G, la.ModularityVertexPartition)
        print(f'[{nnodes}/{p2}]: {time.time() - t0}')
#ig.plot(partition, "todel.png")
'''

def f1(X, Y):
    bbt = BallTree(X, leaf_size=2, metric='euclidean')
    return bbt.query(Y)

def f2(X, Y):
    bbt = KDTree(X, leaf_size=2, metric='euclidean')
    return bbt.query(Y)


X = np.random.random((50, 3))
Y = np.random.random((80000, 3))

n1, n2 = 5, 10
for i in range(n2):
    if i == n1:
        t0 = time.time()
    f1(Y, X)
print(time.time() - t0)
