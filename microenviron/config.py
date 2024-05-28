##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
mRMR_f3 = ['Length', 'AverageContraction', 'AverageFragmentation']
mRMR_f3me = ['Length_me', 'AverageContraction_me', 'AverageFragmentation_me']

BS7_COLORS = {
    'CTX': 'limegreen',
    'CNU': 'darkorange',
    'CB': 'royalblue',
    'TH': 'violet',
    'MB': 'sienna',
    'HY': 'mediumslateblue',
    'HB': 'red'
}

def standardize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

def normalize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.min()) / (fvalues.max() - fvalues.min() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

