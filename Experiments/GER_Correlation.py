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

wahlkreise = ['Berlin-Neukölln', 'Offenburg']
features = ['DIE LINKE', 'AfD']

# wahlkreise = ['Kiel', 'Dresden I']
# features = ['SPD', 'AfD']


dataset.index = dataset.index.to_series().map(names)

# fig, axes = plt.subplots(1,len(coun), figsize=(14,5))

# for cc, ax in zip(coun, axes):
#     val1, val2 = corr

#     ax.scatter(dataset.loc[cc,val1], dataset.loc[cc,val2])
#     ax.set(title=cc,
#            xlabel=val1,
#            ylabel=val2)

#     G = GaussianDistribution()
#     G.estimate(dataset.loc[cc, (val1,val2)])
#     utils.plotGaussian(G, size=0, color='red', STDS=[2], ax=ax)

# # fig.savefig(f"Plots/Correlation_{''.join(coun+corr)}.svg")
# fig.savefig(f"Reports/Figures/GER/Correlation_{''.join(coun+corr)}.pdf")
# plt.show()
# plt.close()

fig, ax = plt.subplots(figsize=(7,7))


for i, wahlkreis in enumerate(wahlkreise):
    val1, val2 = features
    
    color = 'C'+str(i)
    
    correlation = dataset.loc[wahlkreis].corr()
    print(correlation)
    
    ax.scatter(dataset.loc[wahlkreis,val1], 
               dataset.loc[wahlkreis,val2],
               s=10,
               c=color)

    G = GaussianDistribution()
    G.estimate(dataset.loc[wahlkreis, (val1,val2)])
    utils.plotGaussian(G, size=0, color=color, STDS=[2], ax=ax)
    
    x,y = G.mean
    ax.scatter(x,y,label=wahlkreis,c=color,s=200)
    
ax.legend()
ax.set_aspect('equal')    
ax.set(title='Correlation of poll stations within a voting district',
       xlabel=val1,
       ylabel=val2)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

fig.savefig(f"Plots/GER_Correlation_{''.join(features)}.svg")
# fig.savefig(f"Reports/Figures/GER/JointCorrelation_{''.join(coun+corr)}.pdf")
plt.show()
plt.close()