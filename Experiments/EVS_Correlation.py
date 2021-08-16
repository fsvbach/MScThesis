#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:43 2021

@author: bachmafy
"""

import numpy as np
import matplotlib.pyplot as plt
from WassersteinTSNE.Distances import PairwiseWassersteinDistance
from WassersteinTSNE.Distributions import GaussianDistribution
from Experiments.Visualization import utils
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data

counties = ['DK03', 'DK04']
features = ['v143', 'v187']
# corr = ['v103', 'v107']
# corr = ['v199', 'v39']


U = dataset.loc[counties[0], features]
V = dataset.loc[counties[1], features]

_, fig = PairwiseWassersteinDistance(U, V, visualize=True)
fig.savefig("Plots/WassersteinUniformSmart.svg")


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