##########################################################
#Author:          Yufeng Liu
#Create time:     2024-04-28
#Description:     Estimate the feature similarity between
#                 microenvironment and manually annotated
#                 hippocampal neurons from Qiu et al.
##########################################################

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy import stats

from image_utils import image_histeq
from math_utils import get_exponent_and_mantissa

sys.path.append('../..')
from generate_me_map import process_features as process_features_me
from plot_ion_hippocampus import process_features as process_features_ion


def normalize_features(df, keys, norm_type='normalize'):
    # We handling the coloring
    v = df[keys]
    if norm_type == 'normalize':
        df.loc[:, keys] = (v - v.min()) / (v.max() - v.min() + 1e-10)
    elif norm_type == 'standardize':
        df.loc[:, keys] = (v - v.mean()) / v.std()
    elif norm_type == 'histeq':
        for k in keys:
            df.loc[:, k] = image_histeq(df.loc[:, k].values)[0]
    else:
        raise NotImplementedError


# params
me_file = '../../data/mefeatures_100K_with_PCAfeatures3.csv'
ion_file = './ION_HIP/lm_features_d28.csv'
regions = ['CA1', 'CA2', 'CA3', 'ProS', 'SUB', 'DG-mo', 'DG-po', 'DG-sg']
norm_type = 'normalize'
pairing_dist = 3 # in 25um space

# Load the neurons
df_me, keys = process_features_me(me_file, with_comment=False)
df_ion, _ = process_features_ion(ion_file)

# extract only neurons in target regions
df_me = df_me[df_me['region_name_r671'].isin(regions)]
normalize_features(df_me, keys, norm_type)
df_ion = df_ion[df_ion['region_name'].isin(regions)]
normalize_features(df_ion, keys, norm_type)

# find correspondence based on spatial distances in CCFv3 space
ckeys = ['soma_x', 'soma_y', 'soma_z']
neigh = NearestNeighbors(radius=pairing_dist)
neigh.fit(df_me[ckeys])
narr = neigh.radius_neighbors_graph(df_ion[ckeys], pairing_dist)
nids1, nids2 = narr.nonzero()
# plotting
mef = df_me.iloc[nids2][keys].stack().reset_index(1)
ionf = df_ion.iloc[nids1][keys].stack().reset_index(1)
k1, k2, k3 = 'Feature type', 'microenviron', 'Qiu et al., 2024'
df = pd.DataFrame(np.hstack((mef.values, ionf.iloc[:,1].values.reshape(-1,1))), columns=(k1,k2,k3)).astype(
        {k1: str, k2: float, k3: float}
    )

sns.set_theme(style="ticks", font_scale=1.7)
g = sns.lmplot(df, x=k2, y=k3, col=k1, hue=k1, 
               scatter_kws={'s':2},
               facet_kws={'sharex': True, 'sharey': True})

# add annotate
def annotate(data, **kws):
    gg = data[k1].unique()[0]

    r, p = stats.pearsonr(data[k2], data[k3])
    ax = plt.gca()
    ax.text(0.65, 0.16, r'$R={:.2f}$'.format(r),
            transform=ax.transAxes)
    e, m = get_exponent_and_mantissa(p)
    if e is None:
        ax.text(0.65, 0.06, r'$P=0$')
    else:
        ax.text(0.65, 0.06, r'$P={%.1f}x10^{%d}$' % (m, e),
            transform=ax.transAxes)

    ax.set_title(gg)

g.map_dataframe(annotate)
plt.savefig('feat_corr.png', dpi=300)
plt.close()


