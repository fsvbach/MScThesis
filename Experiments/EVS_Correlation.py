#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:43 2021

@author: bachmafy
"""

import matplotlib.pyplot as plt
import numpy as np

from WassersteinTSNE.Distributions import GaussianDistribution
from Experiments.Visualization import Analysis, utils
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data

names = utils.code2name()

coun = ['pl', 'ch']
corr = ['v143', 'v187']
# corr = ['v103', 'v107']

def GaussHistogram(cc, val1, val2):
    H, _, _ = np.histogram2d(dataset.loc[cc, val1], dataset.loc[cc, val2])
    G = GaussianDistribution()
    G.estimate(dataset.loc[cc, (val1, val2)])
    return H, G

# Analysis._config.update(folder='flags', 
#                         seed=13, 
#                         name='NUTS',
#                         description='overview',
#                         dataset='EVS',
#                         renaming= lambda name: EVS.overview[name][0],
#                         size= (3,15),
#                         w=0.5)
# # 
# figure = Analysis.WassersteinEmbedding(dataset, labels)

# A = ['v187', 'v38', 'v103', 'v106', 'v201', 'v107', 'v200', 'v102']
# B = ['v143', 'v39', 'v102', 'v104', 'v188', 'v102', 'v63', 'v186']
# figure = Analysis.Correlations(dataset, labels, selection=zip(A,B), normalize=False, w=1)

dataset.index = dataset.index.to_series().map(labels)

fig, axes = plt.subplots(1,len(coun), figsize=(14,5))

for cc, ax in zip(coun, axes):
    val1, val2 = corr
    H, G = GaussHistogram(cc, val1, val2)

    m = ax.imshow(H.T, cmap='Greens', origin='lower')
    ax.set(title=names[cc.upper()],
           xlabel=EVS.legend[val1][0],
           ylabel=EVS.legend[val2][0])
    plt.colorbar(m, ax=ax)

    utils.plotGaussian(G, size=0, color='red', STDS=[1], ax=ax)

fig.savefig(f"Plots/Correlation_{''.join(coun+corr)}.svg")
fig.savefig(f"Reports/Figures/EVS/Correlation_{''.join(coun+corr)}.pdf")
plt.show()
plt.close()