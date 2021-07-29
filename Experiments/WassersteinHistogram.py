#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:37:54 2021

@author: bachmafy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from WassersteinTSNE.Distances import ConstraintMatrix, EuclideanDistance
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

n = len(P1)

space = np.arange(1,n+1)

D = EuclideanDistance(space.reshape(-1,1), space.reshape(-1,1))

A = ConstraintMatrix(n,n)
b = np.concatenate([P1,P2])
c = D.reshape(-1)
    
opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None], method='highs')

emd = round(opt_res.fun,3)
gamma = opt_res.x.reshape((n, n))


fig, axes = plt.subplots(2,3, figsize=(9,5), 
                         gridspec_kw={'width_ratios': (1,4,4),
                                      'height_ratios': (1,4)})
[ax.set_axis_off() for ax in axes.ravel()]

axes[0,1].bar(space, P1, color='C0', alpha=0.5)
axes[0,1].set(xlim=(0.5,10.5))
axes[1,0].barh(space, P2, color='C1', alpha=0.5)
axes[1,0].set(ylim=(0.5,10.5))
axes[1,0].invert_xaxis()
axes[1,0].invert_yaxis()

axes[1,1].imshow(gamma.T, cmap='Greys', vmin=0)
axes[1,2].imshow(D, cmap='Greys', vmin=0)

axes[0,2].text(0.5,0.5, f"Histogram scipy.linprog EMD={emd}", ha='center')
fig.tight_layout()
fig.savefig("Plots/WassersteinHistogram.svg")
fig.savefig('Reports/Figures/Wasserstein/ExperimentsHistogram.pdf')
plt.show()