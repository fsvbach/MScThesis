#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:57:02 2021

@author: bachmafy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:43 2021

@author: bachmafy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from WassersteinTSNE.Distances import PairwiseWassersteinDistance, GaussianWassersteinDistance
from WassersteinTSNE.Distributions import GaussianDistribution
from Experiments.Visualization import utils
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data

counties = ['DK03', 'DK04']         # 9.936993628081488, 2.0768399363503454
counties = ['AZ-GOR', 'SE32']     # 20.645932405364448, 20.584109832537298

U = dataset.loc[counties[0]]
V = dataset.loc[counties[1]]

# opt_res = PairwiseWassersteinDistance(U, V, visualize=False)
# diff_exact = np.sqrt(-opt_res.fun)

# G = GaussianDistribution()
# Gaussians = [G.estimate(U), G.estimate(V)]

# WSDM = GaussianWassersteinDistance(pd.Series(Gaussians, index=counties))
# diff_gauss = WSDM.matrix()[1,0]
# diff_means = WSDM.matrix(w=0)[1,0]
# diff_covs = WSDM.matrix(w=1)[1,0]  

# print(diff_exact, diff_gauss)
# print(diff_means, diff_covs)

N = len(dataset.columns)
gauss = np.zeros((N,N))
wasst = np.zeros((N,N))
for i, f1 in enumerate(dataset.columns):
    for j, f2 in enumerate(dataset.columns):
        U = dataset.loc[counties[0], [f1,f2]]
        V = dataset.loc[counties[1], [f1,f2]]
        
        G = GaussianDistribution()
        Gaussians = [G.estimate(U), G.estimate(V)]
        WSDM = GaussianWassersteinDistance(pd.Series(Gaussians, index=counties))
        gauss[i,j] = WSDM.matrix()[1,0]
        
        opt_res = PairwiseWassersteinDistance(U, V, visualize=False)
        wasst[i,j] = np.sqrt(-opt_res.fun)

diff = np.abs(wasst-gauss)
plt.imshow(diff)
plt.colorbar()
plt.show()

np.fill_diagonal(diff, 0)
res = np.unravel_index(diff.argmax(), diff.shape)
features = [dataset.columns[idx] for idx in res]

def GaussHistogram(cc, val1, val2):
    H, _, _ = np.histogram2d(dataset.loc[cc, val1], dataset.loc[cc, val2], bins=10)
    G = GaussianDistribution()
    G.estimate(dataset.loc[cc, (val1, val2)])
    return H, G

fig, axes = plt.subplots(1,len(counties), figsize=(14,5))

for cc, ax in zip(counties, axes):
    val1, val2 = features
    H, G = GaussHistogram(cc, val1, val2)

    m = ax.imshow(H.T, cmap='Greens', origin='lower', extent=2*EVS.interval)
    ax.set(title=cc,#names[cc.upper()],
            xlabel=EVS.legend[val1][0],
            ylabel=EVS.legend[val2][0])
    plt.colorbar(m, ax=ax)

    utils.plotGaussian(G, size=0, color='red', STDS=[1], ax=ax)

fig.savefig(f"Plots/EVS_Correlation_{''.join(counties+features)}.svg")
# fig.savefig(f"Reports/Figures/EVS/Correlation_{''.join(coun+corr)}.pdf")
plt.show()
plt.close()