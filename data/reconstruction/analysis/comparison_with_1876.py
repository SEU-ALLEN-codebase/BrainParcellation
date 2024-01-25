import glob
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## Load somalist apo files
def read_apo(folder_path):
    apo_files = glob.glob(folder_path+"/*")
    apo_table = pd.read_csv(apo_files[0])
    for i in range(1, len(apo_files)):
        in_file = pd.read_csv(apo_files[i])
        apo_table = pd.concat([apo_table,in_file])
    apo_table = apo_table.reset_index(drop=True)
    apo_table = apo_table[['name','x','y','z']]
    apo_table['name'] = apo_table['name'].map(lambda x:x.lstrip())
    return apo_table

## match neurons between dataset 1876 & 230k through soma position
def soma_matching(subject_table,object_table):
    matched_ids = []
    sub_ids = []
    for i in range(len(subject_table)):
        s = subject_table[['x','y','z']].iloc[i].values.reshape(1,3)
        d_matrix = distance_matrix(s,object_table[['x','y','z']].values)
        min_d = np.min(d_matrix)
        if min_d > 30:
            continue
        index = np.argwhere(d_matrix == min_d)[0][1]
        matched_ids.append(object_table['name'].iloc[index].split(".swc")[0])
        sub_ids.append(subject_table['name'].iloc[i])
    sub_ids = list(map(lambda x: x.lstrip(),sub_ids))
    matched_ids_obj = list(map(lambda x: x.lstrip(),matched_ids_obj))
    return sub_ids, matched_ids

## extract features in matched order
def ordered_extract(table,ids):
    matched_table = pd.DataFrame()
    for i in ids:
        row = table[table['Name']==i]
        matched_table = pd.concat([matched_table,row])
    matched_table.columns = table.columns
    matched_table = matched_table.reset_index(drop=True)
    return matched_table

## Pearson correlation
def pearson_corr(d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)
    
    c = np.corrcoef(d1.flatten(),d2.flatten())[0,1]
    return c


## Feature table 
gf_table = pd.read_csv('gf_1891_crop.csv')
del_list = ['pre_17543_01171', 'pre_18458_00150', '18454_00008', 'pre_17543_01139', 'pre_17543_01241', 
            '18458_00048', 'pre_17543_01274', 'pre_17543_01276', 'pre_17543_01053', 'pre_17543_01090',
           'pre_18455_00109','17302_00087','18465_00205','18453_3147_x16721_y20510','18453_3218_x16785_y20581']
gf_table['Name'] = gf_table['Name'].map(lambda x:x.split(".swc")[0])
gf_table = gf_table[~gf_table['Name'].isin(del_list)]

gf_folder = 'gf_179k_crop/*'
gf_files = glob.glob(gf_folder)
Ftable=pd.read_csv(gf_files[0])
for i in range(1,len(gf_files)):
    gf_tmp = pd.read_csv(gf_files[i])
    Ftable = pd.concat([Ftable,gf_tmp])
Ftable = Ftable.reset_index(drop=True)
Ftable['Name'] = Ftable['Name'].map(lambda x:x.split("_stps.swc")[0])

folder_subject = "1891_apo"
apo_subject_table = read_apo(folder_subject)
apo_subject_table = apo_subject_table[~apo_subject_table['name'].isin(del_list)]
apo_subject_table = apo_subject_table.reset_index(drop=True)
for i in range(len(apo_subject_table)):
    name = apo_subject_table['name'].iloc[i]
    if '210254' in name:
        new_name = name.replace('210254','15257')
        apo_subject_table.at[i,'name'] = new_name

folder_obj = "179k_somalist_raw"
apo_obj_table = read_apo(folder_obj)

## match 2 datasets
subj_ids, matched_ids_obj = soma_matching(apo_subject_table,apo_obj_table)

matched_sub_gf = ordered_extract(gf_table,subj_ids)
matched_obj_gf = ordered_extract(Ftable,matched_ids_obj)
matched_sub_gf2 = matched_sub_gf.drop(['Name'],axis=1)
matched_obj_gf2 = matched_obj_gf.drop(['Name'],axis=1)

## Pearson 
pearson_c = pearson_corr(matched_sub_gf2,matched_obj_gf2)
print("Pearson correlation:"+str(pearson_c))

## Distribution comparison of six selected feutures 
plt.figure(figsize=(15,8))
feature_list = ['Stems','Bifurcations','Length','MaxPathDistance','MaxBranchOrder','AverageBifurcationAngleLocal']
for i in range(len(feature_list)):
    plt.subplot(2,3,i+1)
    sns.kdeplot(data = matched_sub_gf[feature_list[i]], label="1876 dataset",fill=True)
    sns.kdeplot(data = matched_obj_gf[feature_list[i]], label="auto-reconstructed",fill=True)
    plt.title(feature_list[i],fontsize=15)
    plt.legend()
    plt.tight_layout()
plt.savefig("all_figs/1891_comparison.png",dpi=100)
