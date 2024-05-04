##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-30
#Description:               
##########################################################
mRMR_f3 = ['Length', 'AverageFragmentation', 'AverageContraction']
mRMR_f3me = ['Length_me', 'AverageFragmentation_me', 'AverageContraction_me']

def standardize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

def normalize_features(dfc, feat_names, epsilon=1e-8):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.min()) / (fvalues.max() - fvalues.min() + epsilon)
    dfc.loc[:, feat_names] = fvalues.values

