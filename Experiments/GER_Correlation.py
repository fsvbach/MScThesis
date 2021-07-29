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
from Datasets.GER2017 import Bundestagswahl

GER = Bundestagswahl(numparty=6)
labels  = GER.labeldict()
dataset = GER.data

names = GER.labeldict('Gebiet')

coun = ['Kiel','Dresden I']
corr = ['SPD', 'AfD']

dataset.index = dataset.index.to_series().map(names)

fig, axes = plt.subplots(1,len(coun), figsize=(14,5))

for cc, ax in zip(coun, axes):
    val1, val2 = corr

    ax.scatter(dataset.loc[cc,val1], dataset.loc[cc,val2])
    ax.set(title=cc,
           xlabel=val1,
           ylabel=val2)

    G = GaussianDistribution()
    G.estimate(dataset.loc[cc, (val1,val2)])
    utils.plotGaussian(G, size=0, color='red', STDS=[2], ax=ax)

# fig.savefig(f"Plots/Correlation_{''.join(coun+corr)}.svg")
fig.savefig(f"Reports/Figures/GER/Correlation_{''.join(coun+corr)}.pdf")
plt.show()
plt.close()