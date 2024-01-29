import sys
import numpy as np
from pathlib import Path
from v3dpy.loaders import PBD

    
    
if __name__ == '__main__':
    in_file = Path(sys.argv[1])
    try:
        window = np.array([128, 128, 32])
        img = PBD().load(in_file)[0]
        # thr = img.mean() + .5 * img.std()
        a = (img.shape - window) // 2
        b = (img.shape + window) // 2
        img = img[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
        z, y, x = np.nonzero(img > img.mean() + .5 * img.std())
        coord = np.transpose([z, y, x])
        norm = np.linalg.norm(coord - window // 2, axis=1)
        coord = coord[np.argmin(norm)] + a
    except:
        coord = [512, 512, 128]
    with open(f'{in_file}.marker', 'w') as f:
        f.write('#x, y, z, radius, shape, name, comment,color_r,color_g,color_b\n')
        f.write(f'{coord[2]}, {coord[1]}, {coord[0]}, 5, 1, , , 207, 52, 139')
    # with open('.swc', 'w') as f:
    #     f.write('##n, type, x, y, z, r, p\n')
    #     f.write(f'1 0 {coord[2]} {coord[1]} {coord[0]} 1 -1')
    # with open('thr', 'w') as f:
        # f.write(int(thr))
            
        
