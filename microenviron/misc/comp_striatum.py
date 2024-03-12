##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-11
#Description:               
##########################################################
import numpy as np
import sys
import cv2

from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.ccf_stereotactic_converter import ccf2stereotactic_mask_res25, __RY_CCF25__, __RX_CCF25__, __RZ_CCF25__, __SCALE_Y__

sys.path.append('../')
from parcellation import random_colorize


def random_colorize_mask(mask):
    nzcoords = np.array(mask.nonzero()).transpose()
    values = mask[mask > 0]
    shape3d = mask.shape
    color_level = len(np.unique(values))
    cimg = random_colorize(nzcoords, values, shape3d, color_level)
    return cimg

class StriatumValidator(object):
    def __init__(self, str_indices=(672,56),#,56,998,754),
                bregmas=(1.70,1.10,0.74,0.38,-0.10,-0.58,-1.06,-1.46),
                ):
        self.ccf25m = load_image(MASK_CCF25_FILE)
        self.strm = self.get_striatal_mask(str_indices)
        self.bregmas = np.array(bregmas)

    def get_striatal_mask(self, str_indices):
        strm = np.zeros(self.ccf25m.shape, dtype=bool)
        for istr in str_indices:
            strm = strm | (self.ccf25m == istr)
        return strm

    def get_stereo_mask(self, parc_file='../final_parcellation.nrrd'):
        # load the parcellation
        parc_full = load_image(parc_file)
        # keep only the required regions
        parc = parc_full.copy()
        parc[~self.strm] = 0
        
        stereo_parc = ccf2stereotactic_mask_res25(parc)
        # estimate the sections
        ry = int(np.round(__RY_CCF25__ * __SCALE_Y__))
        rx, rz = __RX_CCF25__, __RZ_CCF25__
        # convert stereotactic coordinates to voxel coordinates
        xs = np.round(self.bregmas * 1000/25.).astype(int) + rx
        # random colorization
        cmask = random_colorize_mask(stereo_parc)
        # extract sections
        sections = np.take(cmask, xs, 2)
        for isec in range(len(xs)):
            sec = np.take(sections, isec, 2)
            sec = cv2.rotate(sec, cv2.ROTATE_90_CLOCKWISE)
            sec = cv2.flip(sec, 1)    # flip horizontal
            # remove non-zero backgrounds
            nzc = sec.sum(axis=-1).nonzero()
            ymax, ymin = nzc[0].max(), nzc[0].min()
            xmax, xmin = nzc[1].max(), nzc[1].min()
            dd = 5
            ymin = max(ymin - dd, 0)
            ymax = min(ymax + dd, sec.shape[0])
            xmin = max(xmin - dd, 0)
            xmax = min(xmax + dd, sec.shape[1])
            sec_sub = sec[ymin:ymax, xmin:xmax]
            cv2.imwrite(f'sec{isec:04d}.png', sec_sub)

        print()


if __name__ == '__main__':
    sv = StriatumValidator()
    sv.get_stereo_mask()

