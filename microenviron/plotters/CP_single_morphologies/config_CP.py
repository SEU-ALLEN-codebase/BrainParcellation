##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-08
#Description:               
##########################################################

SUBREGIONS2COMMU = {
    1: 'CP.ic',
    2: 'CP.r',
    3: 'CP.r',
    4: 'CP.r',
    5: 'CP.i',
    6: 'CP.i',
    7: 'CP.ic',
    8: 'CP.ri',
    9: 'CP.c',
    10: 'CP.ic',
    11: 'CP.ic',
    12: 'CP.i',
    13: 'CP.i'
}

COMMU2SUBREGIONS = {
    'CP.r': [2,3,4],
    'CP.ri': [8],
    'CP.i': [5,6,12,13],
    'CP.ic': [1,7,10,11],
    'CP.c': [9]
}

CCF_ID_CP = 672

