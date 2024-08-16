##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-08
#Description:               
##########################################################

SUBREGIONS2COMMU = {
    1: 'CP.ic',
    2: 'CP.r',
    3: 'CP.ri', # r --> ri
    4: 'CP.r',
    5: 'CP.i',
    6: 'CP.ri', # i --> ri
    7: 'CP.ic',
    8: 'CP.ri',
    9: 'CP.c',
    10: 'CP.i', # ic --> i
    11: 'CP.ic',
    12: 'CP.i',
    13: 'CP.i'
}

COMMU2SUBREGIONS = {
    'CP.r': [2,4],
    'CP.ri': [3,6,8],
    'CP.i': [5,6,10,12,13],
    'CP.ic': [1,7,11],
    'CP.c': [9]
}

COMMU2INDS = {
    'CP.r': 1,
    'CP.ri': 2,
    'CP.i': 3,
    'CP.ic': 4,
    'CP.c': 5
}

INDS2COMMU = {
    1: 'CP.r',
    2: 'CP.ri',
    3: 'CP.i',
    4: 'CP.ic',
    5: 'CP.c'
}

CCF_ID_CP = 672



def get_comm_slice(parc_file='../../output_full_r671/parc_region672.nrrd', sub_mask=True):
    from file_io import load_image
    from image_utils import crop_nonzero_mask

    parc = load_image(parc_file)
    if sub_mask:
        parc,_ = crop_nonzero_mask(parc, pad=1)

    # map subregions to communities
    parc_c = parc.copy()
    for idx, cname in SUBREGIONS2COMMU.items():
        new_id = COMMU2INDS[cname]
        parc_c[parc == idx] = new_id
    # get the section
    slice_lr = parc_c[:,parc_c.shape[1]//2-10]
    return slice_lr

if __name__ == '__main__':
    import cv2
    import numpy as np

    simg = get_comm_slice().astype(np.uint8)
    print(np.unique(simg, return_counts=True))
    print(simg.shape, simg.dtype)
    #simg = (simg / simg.max() * 255).astype(np.uint8)
    cv2.imwrite('cp_slice_lr.png', simg)

