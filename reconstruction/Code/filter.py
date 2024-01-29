import sys
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial

sys.path.insert(0, '/PBshare/SEU-ALLEN/Users/zuohan/pylib')
from file_io import load_image

from skimage.util import img_as_ubyte
from v3dpy.loaders import PBD
from skimage.restoration import denoise_wavelet
from skimage.filters import gaussian


def lateral_filter(img, sigma, truncate=2., suppression=.9):
    diffuse = partial(gaussian, sigma=sigma, preserve_range=True, truncate=truncate)
    gpool = diffuse(img[-1])
    for i in range(img.shape[0]):
        i = img.shape[0] - i - 1
        m = img[i].mean() / gpool.mean() * suppression
        img[i] = (img[i] - m * gpool).clip(0)
        gpool = diffuse(img[i] + gpool)
        

def wavelet_2d(img):
    for i in range(img.shape[0]):
        img[i] = denoise_wavelet(img[i], mode='hard', wavelet_levels=2)


def axial_filter(img):
    mx = img.mean(axis=(0, 1))
    my = img.mean(axis=(0, 2))
    img -= np.add.outer(my, mx)
    
    
if __name__ == '__main__':
    wkdir = Path(sys.argv[0]).parent
    in_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2])
    # out_file = wkdir / 'filtered' / in_file.relative_to(in_file.parents[1]).with_suffix('.v3dpbd')
    # if not out_file.exists():
    out_file.parent.mkdir(exist_ok=True, parents=True)
    
    # params
    brain = int(out_file.parts[-2])
    res_xy = pd.read_csv(wkdir / 'supp.csv', index_col=1, header=0).iloc[:, 3].loc[brain]
    if np.isnan(res_xy):
        res_xy = .25
    sigma = 3 / res_xy
    pct = (.25 / res_xy) ** 2
    rescale_window = np.array([32, 128, 128])
    
    # filtering, with minimum memory copy
    img = load_image(in_file, flip_tif=False).astype(np.float32)  # due to previous bug
    lateral_filter(img, sigma)
    axial_filter(img)
    thr = np.percentile(img, 100 - pct)
    a = (img.shape - rescale_window) // 2
    b = (img.shape + rescale_window) // 2
    levels = min(255, (img[a[0]:b[0], a[1]:b[1], a[2]:b[2]].max() - thr) * .5)
    img -= thr
    np.clip(img, 0, levels, img)
    img /= levels
    wavelet_2d(img)
    img = img_as_ubyte(img.clip(0, 1))
    
    PBD().save(out_file, img.reshape(1, *img.shape))
