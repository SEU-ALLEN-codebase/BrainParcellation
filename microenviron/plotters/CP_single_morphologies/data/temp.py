##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-15
#Description:               
##########################################################
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

neuron_file = 'match_matrix.csv'
neurons = pd.read_csv(neuron_file, header=None).astype(int)
# row-wise normalization
neurons = neurons / (neurons.values.sum(axis=1).reshape((-1,1))+1e-10) * 100

# customize colormap
reds = plt.cm.get_cmap('Reds')
# Create a new colormap by rescaling the 'Reds' colormap to make it brighter
bright_reds = LinearSegmentedColormap.from_list('bright_reds', reds(np.linspace(0, 0.75, 100)))
sns.heatmap(neurons, cmap=bright_reds)

hm = np.zeros_like(neurons)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        # get the 2x2 contingency table
        a = neurons.iloc[i,j]
        b = neurons.iloc[i].sum() - a
        c = neurons.iloc[:,j].sum() - a
        d = neurons.sum().sum() - a - b - c
        ctable = np.array([[a,b],[c,d]])
        # do one-sided fisher exact test
        s, p = fisher_exact(ctable, alternative='greater')
        #print(s, p)
        if p < 0.0001:
            hm[i,j] = 1
        else:
            hm[i,j] = 0
data = pd.DataFrame(np.array(np.nonzero(hm)).transpose(), columns=('x', 'y'))
data += 0.5
sns.scatterplot(data, x='y', y='x', marker='*')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('temp.png', dpi=300)
plt.close()

