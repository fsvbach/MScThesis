#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:25:16 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data

feature = 'v102'
dataset.index = dataset.index.to_series().map(labels)

P1 = np.histogram(dataset.loc['de', feature][:60], bins=10, density=False)
P2 = np.histogram(dataset.loc['al', feature][:40], bins=10, density=False)

P1 = P1[0]/P1[0].sum()
P2 = P2[0]/P2[0].sum()

minval, maxval = EVS.interval
space, step  = np.linspace(minval,  maxval, 10, retstep=True)
dist = round(wasserstein_distance(space, space, P1, P2),3)

fig, ax = plt.subplots()

ax.bar(space, P1, width=0.9*step, alpha=0.5, label='Germany')
ax.bar(space, P2, width=0.9*step, alpha=0.5, label='Albania')

ax.set(xticks=np.round(space,2),
       title =rf'scipy.wasserstein\_distance EMD={dist} (only 1D)')
ax.legend()

fig.savefig('Plots/WassersteinExperiments1D.svg')
fig.savefig('Reports/Figures/Wasserstein/Experiments1D.pdf')
plt.show()
plt.close()