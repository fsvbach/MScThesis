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

P1 = np.histogram(dataset.loc['de', feature], bins=10, density=False)
P2 = np.histogram(dataset.loc['al', feature], bins=10, density=False)

P1 = P1[0]/P1[0].sum()
P2 = P2[0]/P2[0].sum()

space = np.arange(1,11)
dist = round(wasserstein_distance(space, space, P1, P2),2)

fig, ax = plt.subplots()

ax.bar(space, P1, alpha=0.5, label='Germany')
ax.bar(space, P2, alpha=0.5, label='Albania')

ax.set(xticks=space,
       title =f'Two 1D distributions with EMD={dist} ({feature})')
ax.legend()

fig.savefig('Plots/WassersteinExperiments1D.svg')
plt.show()
plt.close()