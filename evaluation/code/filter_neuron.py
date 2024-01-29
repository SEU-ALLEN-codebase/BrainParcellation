import pandas as pd
import numpy as np

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


def cal_criteria_subj(table,type_involved,features):
    criteria_table = np.zeros((len(type_involved),2*len(features)))
    min_max = np.zeros((len(features),2))
    for f in features: 
        f_id = features.index(f)
        for t in type_category:
            selected = np.array(table[table['soma_region']==t][f])
            if (len(selected) < 2)|(t=='unknown'):
                continue
            tid = type_involved.index(t)
            lower = np.min(selected)
            upper = np.max(selected)
            criteria_table[tid,2*f_id] = lower
            criteria_table[tid,2*f_id+1] = upper
            #print(f+' '+t+' '+str(lower)+' '+str(upper))
    
    ## min-max standard
    for i in range(len(features)):
        min_max[i,0] = table[features[i]].min()   
        min_max[i,1] = table[features[i]].max()
        
    return criteria_table,min_max

def filter_neuron(table,features,type_involved,criteria_table,min_max):
    filtered_table = table[table['reassigned_type']!='unknown']
    
    filtered_swc_list = []
    for i in type_involved:
        rows = filtered_table[filtered_table['reassigned_rough_type'] == i] 
        if len(rows) == 0:
            continue
        tid = type_involved.index(i)
        j = 0
        s = rows.copy()
        while(j<len(features)):
            s = s[(s[features[j]]>=criteria_table[tid,2*j]*0.95)&(s[features[j]]<=criteria_table[tid,2*j+1]*1.05)]
            j=j+1
        if len(s) > 0:
            filtered_swc_list = filtered_swc_list+list(s['Name'])
    
    rest = filtered_table[~filtered_table['reassigned_rough_type'].isin(type_involved)]
    j = 0
    if len(rest) != 0:
        s = rest.copy()
        while(j<len(features)):
            s = s[(s[features[j]]>=min_max[j,0]*0.95)&(s[features[j]]<=min_max[j,1]*1.05)]
            j=j+1
        if len(s) > 0:
            filtered_swc_list = filtered_swc_list+list(s['Name'])
    return filtered_swc_list


## subject(1876 dataset) statistics
type_category = ['TH','CTX','CNU','HY','CB','MB','HB','fiber','unknown']
type_involved = ['TH','CTX','CNU','HB']
features = ['Length','Bifurcations']

gf_subj_path = 'gf_1876_crop.csv'
gf_subj_table = pd.read_csv(gf_subj_path)
celltype_table_path = "/home/penglab/Desktop/MyFiles/Projects/1876_Soma_region.csv"
celltype_table = pd.read_csv(celltype_table_path)
full_subj_table = pd.merge(gf_subj_table,celltype_table,on='Name')
for i in type_category:
    full_subj_table.loc[full_subj_table['soma_region'].isin(eval(i)),'rough_type'] = i
criteria_table,min_max = cal_criteria_subj(full_subj_table,type_involved,features)

## screening
gf_obj_path = 'gf_179k_crop.csv'
gf_obj_table = pd.read_csv(gf_obj_path)
celltype_obj_path = 'reassigned_table.csv'
celltype_obj = pd.read_csv(celltype_obj_path)
full_obj_table = pd.merge(gf_obj_table,celltype_obj,on='Name')
filtered_swc_confirmed = filter_neuron(full_obj_table,features,type_involved,criteria_table,min_max)
      
filtered_confirmed_table = full_obj_table[full_obj_table['Name'].isin(filtered_swc_confirmed)][['Name','reassigned_type','reassigned_rough_type']]
filtered_confirmed_table.to_csv('filtered_swc.csv',index=None)