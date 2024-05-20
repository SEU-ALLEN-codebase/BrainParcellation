##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-19
#Description:               
##########################################################
import numpy as np

def crop_regional_mask(mask, v=None):
    if type(v) is int:
        reg_mask = mask == v
    elif type(v) is list:
        reg_mask = np.zeros_like(mask).astype(bool)
        for vi in v:
            reg_mask[mask == vi] = True
    else:
        reg_mask = mask > 0
    
    nzcoords_t = np.array(reg_mask.nonzero()).transpose()
    
    zmin, ymin, xmin = nzcoords_t.min(axis=0)
    zmax, ymax, xmax = nzcoords_t.max(axis=0)
    sub_mask = mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    return sub_mask, zmin, zmax, ymin, ymax, xmin, xmax

