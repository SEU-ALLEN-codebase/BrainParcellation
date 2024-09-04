import pickle
import pandas as pd
import numpy as np
import pandas as pd
import sys
import os
#from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
sys.path.append('../../neuro_morpho_toolbox_20200908')
import neuro_morpho_toolbox as nmt


## Brain region categorization
TH=['AD','AM','AV','CL','CM','IAD','LD','LGd','LHA','LP','MD','MG','MM','PCN','PF','PIL','PO','POL','PR','PVT',
   'PoT','RE','RT','SGN','SMT','TH','VAL','VM','LGv','IMD','PT','SPA','SPFp','LH','MH','SPFm','PP','RH','IGL','IntG',
   'VPL','VPLpc','VPM','VPMpc','SubG','AMd','Xi','IAM']
CTX=['ACAd','ACAv','AId','AIp','AIv','AON','AON','AUDd','AUDpo','AUDv','BLA','BMA','CA1','CA3','CLA','COAp','HPF','DG',
    'ECT','ENTl','ENTm','EPd','FRP','GU','IG','ILA','MOB','MOp','MOs','ORBl','ORBm','ORBvl','PIR','PL','POST','PRoS',
    'RSPagl','RSPd','RSPv','SSs','SUB','TEa','TR','VISC','VISa','VISal','VISam','VISl','VISli','VISp','VISpm',
    'VISpor','VISrl','VISpl','PAR','SSp','SSp-bfd','SSp-ll','SSp-m','SSp-n','SSp-tr','SSp-ul','SSp-un','AOB','APr',
    'DP','AUDp','TT','ProS','LA','PERI','NLOT','COAa','CA2','HATA','EPv','DG-sg','DG-mo','DG-po','ENTm6',
    'BMAa','COApl','BMAp','ENTl2','ENTl1','OLF','ENTm3','PRE','PAA','PA']
## CTXsp: PA
CNU=['ACB','BST','CEA','CP','FS','GPe','GPi','LSr','LSv','OT','PAL','SI','MEA','MS','LSc','AAA','MA','NDB',
    'TRS','SF','IA','SH','BA']
## PAL: TRS
## STR: SF,IA,SH,BA
HY=['AHN','DMH','HY','PH','PSTN','PVH','PVi','PeF','SBPV','TU','VMH','ZI','SUM','STN','MMd','MMme','MMl',
   'ARH','PMv','PVp','TMv','LM','LPO','PS','MPN','MPO','MEPO','ADP','PMd','AVPV','SCH',
   'PVpo','PVHd','VLPO','VMPO','ME','RCH','PVa','SO']
CB=['CENT','COPY','CUL','DEC','FL','FN','FOTU','NOD','PFL','AN','PRM','PYR','SIM','SIM','CENT3','CUL4, 5','CENT2',
   'ANcr2','ANcr1','UVU','LING','IP','VeCB','DN']
## CBX: UVU,LING
## CBN:IP,VeCB,DN
MB=['DR','IC','MB','MRN','MV','MY','PAG','RN','SNc','SCm','APN','PPN','RR','SNr','SAG','SCs','VTN','NOT','NB','AT',
   'CUN','PPT','MPT','VTA','CLI','NPC','IPN','IF','SCO','PN','ICe','LT','OP','PBG','EW',
   'MA3','RL','MT','III','DT']
HB=['IRN','LRN','MDRNd','MDRNv','P','PARN','PB','PG','PGRNl','PRNc','PRNr','PSV','SPIV','SPVC','SPVO','SUV','TRN',
   'V','VCO','NLL','CS','RPO','PCG','LDT','PCG','DTN','SPVI','LRNm','NI','KF','IO','RPA',
   'MARN','SOC','GRN','VII','PPY','AMB','LIN','RM','NTB','NR','XII','NTS','PGRNd','CU','P5',
   'GR','DMX','Pa5','AP','PAS','x','ECU','LAV','LC','I5','DCO','RO','SLD','PRP','SUT','ISN',
   'PC5','VI','B']
## MY:IO,RPA,MARN,GRN,VII,PPY,AMB,x,ECU,LIN,RM,LAV,NTB,NR,XII,NTS,PGRNd,CU,GR,DMX,Pa5,AP,PAS,
## DCO,RO,PRP,ISN,VI
## P:SOC,P5,LC,I5,SLD,SUT,PC5,B
fiber=['fi','dhc','fr','fp','alv','cing','sm','mtt','pm','nst','or','fa','ccg','arb','mlf','ll','fx','int',
      'ccb','mfb','tspc','arb','ml','scwm','fiber tracts']
unknown=['unknown']
error=['error']

type_category = ['TH','CTX','CNU','HY','CB','MB','HB','fiber','unknown','error']

ccf_reassigned_f='/home/penglab/Desktop/MyFiles/neuro_morpho_toolbox_20200908/ccf_reassigned.pickle'


def brain_region_matrix(my_annotation,regionList,brain_region_folder):
    brainRegion_range = pd.DataFrame(columns=['region','xmin','xmax','ymin','ymax','zmin','zmax'])
    ccf_z_mid=nmt.annotation.size['z']/2
    
    for region in regionList:
        if region in ['MDRN','fiber tracts','SSp','ADUv','BS']:
            continue
        selected_annotaion=my_annotation.array
        region_id=nmt.bs.name_to_id(region)
        
        ix, iy, iz = np.where(selected_annotaion == region_id)
        ix_min=ix.min()
        ix_max=ix.max()
        iy_min=iy.min()
        iy_max=iy.max()
        iz_min=iz.min()
        iz_max=0
        for zi in np.arange(iz.size):
            if iz[zi] < ccf_z_mid and iz_max < iz[zi]:
                iz_max = iz[zi]
        brainRegion_range = brainRegion_range._append({'region':region,'xmin':ix_min,'xmax':ix_max,'ymin':iy_min,'ymax':iy_max,
                                                  'zmin':iz_min,'zmax':iz_max},ignore_index=True)
        
        saved_region_f=brain_region_folder+region+'_boundary.pickle'
        if os.path.exists(saved_region_f):
            continue

        boundary_data=[]
        for x in np.arange(ix_min,ix_max+1):
            for y in np.arange(iy_min,iy_max+1):
                for z in np.arange(iz_min,iz_max+1):
                    if selected_annotaion[x,y,z] != region_id:
                        continue
                    boundary=False
                    for kx in [-1,0,1]:
                        for ky in [-1,0,1]:
                            for kz in [-1,0,1]:
                                if selected_annotaion[x+kx,y+ky,z+kz] != region_id:
                                    boundary=True
                                    break
                    if boundary:
                        boundary_data.append([x,y,z])
        
        with open(saved_region_f,'wb') as f:
            pickle.dump(boundary_data,f)
    
    return brainRegion_range


def reassign_region(in_parameters):
    f_neuron_index = in_parameters[0]
    s_info = in_parameters[1]
    brainR_range = in_parameters[2]
    
    soma_xyz = s_info[['x','y','z']].values[0]
    soma_xyz = soma_xyz/25
    
    n = 0
    expansion_Round = 0
    #interval = [0,0.57,1.15]
    interval = [i for i in range(10)]
    while ((n == 0)&(expansion_Round<5)):      
        soma_xyz_large = soma_xyz+interval[expansion_Round]
        soma_xyz_small = soma_xyz-interval[expansion_Round]
        rows = brainR_range[((brainR_range['xmax']>=soma_xyz_small[0])&(brainR_range['xmin']<=soma_xyz_large[0]))|
                                  ((brainR_range['ymax']>=soma_xyz_small[1])&(brainR_range['ymin']<=soma_xyz_large[1]))|
                                   ((brainR_range['zmax']>=soma_xyz_small[2])&(brainR_range['zmin']<=soma_xyz_large[2]))]
        n = len(rows)
        expansion_Round = expansion_Round + 1

    soma_int = [round(soma_xyz[0]),round(soma_xyz[1]),round(soma_xyz[2])]
    
    minD = []
    reR = []
    
    if (len(rows)==0):
        return [f_neuron_index,'unknown']
        
    for j in list(rows['region']):
        boundary_file = "region_boundary/"+str(j)+"_boundary.pickle"
        pickle_file = open(boundary_file, 'rb')
        region_content = pickle.load(pickle_file)
        region_content_df = pd.DataFrame(region_content, columns=['x', 'y','z'])  
        region_XYZ = region_content_df[['x','y','z']].values
        d = np.linalg.norm(region_XYZ - soma_xyz,axis=1)
        min_d = np.min(d)
        
        match = region_content_df[(region_content_df['x']==soma_int[0])&(region_content_df['y']==soma_int[1])&
                                   (region_content_df['z']==soma_int[2])]
        if len(match) > 0:
            return [i,j]
        else:
            minD.append(min_d)
            reR.append(j)
    minD = np.array(minD)
    min_dist = np.min(minD)
    min_index = np.where(minD==min_dist)[0].tolist()
    if(min_dist<=2):
        if len(min_index) == 1:
            R = reR[min_index[0]]
            return [f_neuron_index,R]
        else:
            return [f_neuron_index,'unknown']
    else:
        return [f_neuron_index,'unknown']


def run_reassignment(table,neuron_type,soma_info,brainRegion_range,cpu_workder_num):
    neurons = table[table['Soma region']==neuron_type]
    neuron_index = list(table[table['Soma region']==neuron_type].index)
    parameters = []
    for i in range(len(neurons)):
        name = neurons.iloc[i]['Name']
        s_info = soma_info[soma_info['name']==name]
        in_para = [neuron_index[i],s_info,brainRegion_range]
        parameters.append(in_para)
        
    with Pool(cpu_workder_num) as pool:
        results = pool.map(reassign_region,parameters)
    pool.close()
    
    return results



## generate CCFv3 img matrix for each brain region
region_table = pd.read_excel("CCFv3 Summary Structures.xlsx")
regionList = np.unique(region_table['abbreviation'])
brain_region_folder = "./region_boundary/"
ccf_reassigned_f='/home/penglab/Desktop/MyFiles/neuro_morpho_toolbox_20200908/ccf_reassigned.pickle'
with open(ccf_reassigned_f,'rb') as f:
    my_annotation=pickle.load(f)
brainRegion_range = brain_region_matrix(my_annotation,regionList,brain_region_folder)


## 179k information table
table = pd.read_csv("179k_local_tracing.csv")
soma_info = pd.read_csv('179k_somalist.txt',sep=' ',header=None)
soma_info.columns=['name','x','y','z']
soma_info['name'] = soma_info['name'].map(lambda x:x.split("_stps")[0])
type_table_rough = table.copy()
for i in type_category:
    type_table_rough.loc[type_table_rough['Soma region'].isin(eval(i)),'rough_type'] = i
type_table_rough['Name'] = type_table_rough['Name'].map(lambda x:x.split("_stps")[0])
### fiber tracts(10769) & error neurons(23702)
results_fiber = run_reassignment(type_table_rough,'fiber tracts',soma_info,brainRegion_range,15)
np.savetxt('fiber.txt',np.array(results_fiber),fmt='%s')
results_error = run_reassignment(type_table_rough,'error',soma_info,brainRegion_range,15)
np.savetxt('error.txt',np.array(results_error),fmt='%s')
### map back
reassigned_type = list(type_table_rough['Soma region'].copy())
for i in range(len(results_fiber)):
    reassigned_type[results_fiber[i][0]] = results_fiber[i][1]
for i in range(len(results_error)):
    reassigned_type[results_error[i][0]] = results_error[i][1]
type_table_rough['reassigned_type'] = reassigned_type
for i in type_category:
    type_table_rough.loc[type_table_rough['reassigned_type'].isin(eval(i)),'reassigned_rough_type'] = i

out_table = type_table_rough[['Name','Soma region','reassigned_type','reassigned_rough_type']]   
out_table.to_csv('reassigned_table.csv',index=None) 
