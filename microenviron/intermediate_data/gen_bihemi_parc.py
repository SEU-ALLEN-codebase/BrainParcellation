##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-06
#Description:               
##########################################################
import numpy as np
import pickle
from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE, SALIENT_REGIONS

# convert the ccf-me atlas to bihemispheric
parc_file = 'parc_r671_full.nrrd'
out_file = 'parc_r671_full_hemi2.nrrd'

parc = load_image(parc_file)
ccf = load_image(MASK_CCF25_FILE)
# flip
zdim = 456
z2 = zdim // 2
parc[:z2] = np.flip(parc, 0)[:z2]
save_image(out_file, parc)
# check the symmetry in CCF
for sreg in SALIENT_REGIONS:
    # extract the left and right mask in CCF
    m_ccf = ccf == sreg
    lm_ccf = m_ccf.copy(); lm_ccf[z2:] = 0
    rm_ccf = m_ccf.copy(); rm_ccf[:z2] = 0
    # check the 
    #m_parc = parc == sreg
    #lm_parc = m_parc[:z2]
    #rm_parc = m_parc[z2:]
    # check
    print(f'---> [Region: ] sreg')
    if lm_ccf.sum() != rm_ccf.sum():
        print(f'No. of voxels in region {sreg} is not symmetry: left/all={100.*lm_ccf.sum()/(lm_ccf.sum()+rm_ccf.sum()):.2f}')
        continue

    luniq = np.unique(parc[lm_ccf])
    runiq = np.unique(parc[rm_ccf])
    if luniq.shape[0] != runiq.shape[0]:
        print(f'No. of subregions across left and right not the same: [left={luniq}] vs. [right={runiq}]')
        continue

    if (luniq == runiq).all():
        print(f'Left-right subregions are not identical: [left={luniq}] vs. [right={runiq}]')
        continue



