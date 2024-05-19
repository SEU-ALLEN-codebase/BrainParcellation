##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-16
#Description:               
##########################################################

import os
import glob
import sys
import subprocess
import time
import pandas as pd


def resample_swc(swc_in, swc_out, step=2, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x', correction=True):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x resample_swc -f resample_swc -i {swc_in} -o {swc_out} -p {step}'
    print(swc_in, swc_out)
    p = subprocess.check_output(cmd_str, shell=True)
    if correction:
        # The built-in resample_swc has a bug: the first node is commented out, and there are two additional columns
        subprocess.run(f"sed -i 's/pid1/pid\\n1/g' {swc_out}; sed -i 's/ -1 -1//g' {swc_out}", shell=True)
    return True

def sort_swc(swc_in, swc_out=None, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x'):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x sort_neuron_swc -f sort_swc -i {swc_in} -o {swc_out}'
    p = subprocess.check_output(cmd_str, shell=True)

    # retype
    df = pd.read_csv(swc_out, sep=' ', names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'), comment='#', index_col=False)
    df['type'] = 3
    df.loc[0, 'type'] = 1
    df.to_csv(swc_out, sep=' ', index=False)

    return True

def resample_sort_swc(swc_in, swc_out):
    resample_swc(swc_in, swc_out)
    sort_swc(swc_out, swc_out)


if __name__ == '__main__':
    indir = '/PBshare/SEU-ALLEN/Users/Sujun/230k_organized_folder/cropped_100um'
    outdir = '/data/lyf/data/200k_v2/cropped_100um_resampled2um'

    
    args_list = []
    for bdir in glob.glob(os.path.join(indir, '[1-9]*[0-9]')):
        bid = os.path.split(bdir)[-1]
        obdir = os.path.join(outdir, bid)
        if not os.path.exists(obdir):
            os.mkdir(obdir)
        # process each brain
        for swcfile in glob.glob(os.path.join(bdir, '*.swc')):
            fn = os.path.split(swcfile)[-1]
            oswcfile = os.path.join(obdir, fn)
            if not os.path.exists(oswcfile):
                args_list.append((swcfile, oswcfile))


    # multiprocessing
    from multiprocessing import Pool
    pool = Pool(processes=12)
    pool.starmap(resample_sort_swc, args_list)
    pool.close()
    pool.join()
