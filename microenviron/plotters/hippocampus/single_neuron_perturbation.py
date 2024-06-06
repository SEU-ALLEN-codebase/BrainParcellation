##########################################################
#Author:          Yufeng Liu
#Create time:     2024-06-06
#Description:               
##########################################################
import random
import copy
import numpy as np
import pandas as pd
import seaborn as sns

from swc_handler import parse_swc, write_swc, prune, find_soma_node

class MorphologyPerturbation:
    def __init__(self, swcfile, seed=1024):
        self.tree = parse_swc(swcfile)
        self.nsids = [i for i, node in enumerate(self.tree) if node[-6] != -1]
        self.sid = find_soma_node(self.tree)
        # deterministic
        random.seed(seed)
        np.random.seed(seed)

    def random_remove_points(self, npoints, outswcfile=None):
        sel_ids = random.sample(self.nsids, npoints)
        sel_indices = [self.tree[sel_id][0] for sel_id in sel_ids]
    
        # remove disconnected nodes
        pruned = prune(self.tree, set(sel_indices))
        #print(len(self.tree), len(pruned))
        if outswcfile is not None:
            write_swc(pruned, outswcfile)

        return pruned


if __name__ == '__main__':
    swcfile = '/PBshare/SEU-ALLEN/Users/Sujun/ION_Hip_CCFv3_crop/202268_015.swc'
    outswcfile = 'tmp.swc'
    
    mp = MorphologyPerturbation(swcfile)
    pruned = mp.random_remove_points(5)
    print(pruned[:10])
        
