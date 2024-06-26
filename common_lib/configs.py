


__FEAT_NAMES__ = [
    'Stems', 'Bifurcations',
    'Branches', 'Tips', 'OverallWidth', 'OverallHeight', 'OverallDepth',
    'Length', 'Volume',
    'MaxEuclideanDistance', 'MaxPathDistance', 'MaxBranchOrder',
    'AverageContraction', 'AverageFragmentation',
    'AverageParent-daughterRatio', 'AverageBifurcationAngleLocal',
    'AverageBifurcationAngleRemote', 'HausdorffDimension']


__FEAT_ALL__ = [
    'Stems_mean', 'Bifurcations_mean',
    'Branches_mean', 'Tips_mean', 'OverallWidth_mean', 'OverallHeight_mean',
    'OverallDepth_mean', 'Length_mean',
    'Volume_mean', 'MaxEuclideanDistance_mean', 'MaxPathDistance_mean',
    'MaxBranchOrder_mean', 'AverageContraction_mean', 'AverageFragmentation_mean',
    'AverageParent-daughterRatio_mean', 'AverageBifurcationAngleLocal_mean',
    'AverageBifurcationAngleRemote_mean', 'HausdorffDimension_mean',
    'Stems_median', 'Bifurcations_median',
    'Branches_median', 'Tips_median', 'OverallWidth_median', 'OverallHeight_median',
    'OverallDepth_median', 'Length_median',
    'Volume_median', 'MaxEuclideanDistance_median', 'MaxPathDistance_median',
    'MaxBranchOrder_median', 'AverageContraction_median', 'AverageFragmentation_median',
    'AverageParent-daughterRatio_median', 'AverageBifurcationAngleLocal_median',
    'AverageBifurcationAngleRemote_median', 'HausdorffDimension_median',
    'Stems_std', 'Bifurcations_std',
    'Branches_std', 'Tips_std', 'OverallWidth_std', 'OverallHeight_std',
    'OverallDepth_std', 'Length_std',
    'Volume_std', 'MaxEuclideanDistance_std', 'MaxPathDistance_std',
    'MaxBranchOrder_std', 'AverageContraction_std', 'AverageFragmentation_std',
    'AverageParent-daughterRatio_std', 'AverageBifurcationAngleLocal_std',
    'AverageBifurcationAngleRemote_std', 'HausdorffDimension_std'
    ]

# Brain regions
REGION314_D = {
    'TH':['AD','AM','AV','CL','CM','IAD','LD','LGd','LHA','LP','MD','MG',
        'MM','PCN','PF','PIL','PO','POL','PR','PVT','PoT','RE','RT','SGN',
        'SMT','TH','VAL','VM','LGv','IMD','PT','SPA','SPFp','LH','MH',
        'SPFm','PP','RH','IGL','IntG','VPL','VPLpc','VPM','VPMpc','SubG',
        'AMd','Xi','IAM'],
    'CTX':['ACAd','ACAv','AId','AIp','AIv','AON','AON','AUDd','AUDpo','AUDv',
         'BLA','BMA','CA1','CA3','CLA','COAp','HPF','DG','ECT','ENTl','ENTm',
         'EPd','FRP','GU','IG','ILA','MOB','MOp','MOs','ORBl','ORBm','ORBvl',
         'PIR','PL','POST','PRoS','RSPagl','RSPd','RSPv','SSs','SUB','TEa',
         'TR','VISC','VISa','VISal','VISam','VISl','VISli','VISp','VISpm',
         'VISpor','VISrl','VISpl','PAR','SSp','SSp-bfd','SSp-ll','SSp-m',
         'SSp-n','SSp-tr','SSp-ul','SSp-un','AOB','APr','DP','AUDp','TT',
         'ProS','LA','PERI','NLOT','COAa','CA2','HATA','EPv','DG-sg','DG-mo',
         'DG-po','ENTm6','BMAa','COApl','BMAp','ENTl2','ENTl1','OLF','ENTm3',
         'PRE','PAA','PA'],
    ## CTXsp: PA
    'CNU':['ACB','BST','CEA','CP','FS','GPe','GPi','LSr','LSv','OT','PAL','SI',
         'MEA','MS','LSc','AAA','MA','NDB','TRS','SF','IA','SH','BA'],
    ## PAL: TRS
    ## STR: SF,IA,SH,BA
    'HY':['AHN','DMH','HY','PH','PSTN','PVH','PVi','PeF','SBPV','TU','VMH','ZI',
        'SUM','STN','MMd','MMme','MMl','ARH','PMv','PVp','TMv','LM','LPO','PS',
        'MPN','MPO','MEPO','ADP','PMd','AVPV','SCH','PVpo','PVHd','VLPO','VMPO',
        'ME','RCH','PVa','SO'],
    'CB':['CENT','COPY','CUL','DEC','FL','FN','FOTU','NOD','PFL','AN','PRM','PYR',
        'SIM','SIM','CENT3','CUL4, 5','CENT2','ANcr2','ANcr1','UVU','LING','IP',
        'VeCB','DN'],
    ## CBX: UVU,LING
    ## CBN:IP,VeCB,DN
    'MB':['DR','IC','MB','MRN','MV','MY','PAG','RN','SNc','SCm','APN','PPN','RR',
        'SNr','SAG','SCs','VTN','NOT','NB','AT','CUN','PPT','MPT','VTA','CLI',
        'NPC','IPN','IF','SCO','PN','ICe','LT','OP','PBG','EW','MA3','RL','MT',
        'III','DT'],
    'HB':['IRN','LRN','MDRNd','MDRNv','P','PARN','PB','PG','PGRNl','PRNc','PRNr',
        'PSV','SPIV','SPVC','SPVO','SUV','TRN','V','VCO','NLL','CS','RPO','PCG',
        'LDT','PCG','DTN','SPVI','LRNm','NI','KF','IO','RPA','MARN','SOC','GRN',
        'VII','PPY','AMB','LIN','RM','NTB','NR','XII','NTS','PGRNd','CU','P5',
        'GR','DMX','Pa5','AP','PAS','x','ECU','LAV','LC','I5','DCO','RO','SLD',
        'PRP','SUT','ISN','PC5','VI','B'],
    ## MY:IO,RPA,MARN,GRN,VII,PPY,AMB,x,ECU,LIN,RM,LAV,NTB,NR,XII,NTS,PGRNd,CU,GR,DMX,Pa5,AP,PAS,
    ## DCO,RO,PRP,ISN,VI
    ## P:SOC,P5,LC,I5,SLD,SUT,PC5,B
}
REG_FIBERS=['fi','dhc','fr','fp','alv','cing','sm','mtt','pm','nst','or','fa',
            'ccg','arb','mlf','ll','fx','int','ccb','mfb','tspc','arb','ml',
            'scwm','fiber tracts']


