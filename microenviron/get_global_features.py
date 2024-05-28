##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-17
#Description:               
##########################################################
import os
import sys
import glob
import pandas as pd

from global_features import calc_global_features_from_folder


def get_features_for_brains(swc_dir):
    args_list = []
    for bdir in glob.glob(os.path.join(swc_dir, '[1-9]*[0-9]')):
        bid = os.path.split(bdir)[-1]
        outfile = f'{bid}.csv'
        if os.path.exists(outfile):
            continue
        args_list.append((bdir, outfile))

    print(f'Number of brains to calculate: {len(args_list)}')

    # multiprocessing
    from multiprocessing import Pool
    pool = Pool(processes=16)
    pool.starmap(calc_global_features_from_folder, args_list)
    pool.close()
    pool.join()

def merge_all_brains(csv_dir):
    for i, csv_file in enumerate(sorted(glob.glob(os.path.join(csv_dir, '*.csv')))):
        print(i, os.path.split(csv_file)[-1])
        dfi = pd.read_csv(csv_file, index_col=None)
        if i == 0:
            df = dfi
        else:
            df = pd.concat([df, dfi], ignore_index=True)
        
    print(df.shape)
    df.rename({'Unnamed: 0': 'Name'}, axis=1, inplace=True)
    df.to_csv('gf_179k_crop_resampled.csv', index=False)
        

if __name__ == '__main__':
    #get_features_for_brains('/data/lyf/data/200k_v2/cropped_100um_resampled2um')
    
    #merge_all_brains('global_features')

    # for gold standard
    calc_global_features_from_folder('../evaluation/data/1891_100um_2um_dendrite', '../evaluation/data/gf_1876_crop_2um_dendrite.csv')

