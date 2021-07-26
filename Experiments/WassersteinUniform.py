#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:37:54 2021

@author: bachmafy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from WassersteinTSNE.Distances import linprogSolver, EuclideanDistance, ConstraintMatrix
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data

feature = 'v102'
dataset.index = dataset.index.to_series().map(labels)

U = dataset.loc['de', feature][:60].values.reshape(-1,1)
V = dataset.loc['al', feature][:40].values.reshape(-1,1)

n,m = len(U), len(V)

D = EuclideanDistance(U, V)

A = ConstraintMatrix(n,m)
b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
c = D.reshape(-1)
    
opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None], method='highs')

# opt_res = linprogSolver(U,V)

emd = round( opt_res.fun,3)
gamma = opt_res.x.reshape((n, m))

fig, axes = plt.subplots(2,3, figsize=(13,5), 
                         gridspec_kw={'width_ratios': (1,6,6),
                                      'height_ratios': (1,4)})
[ax.set_axis_off() for ax in axes.ravel()]

axes[0,1].bar(np.arange(n), 1, color='C0', alpha=0.5)
axes[0,1].set(xlim=(-0.5,n-0.5))
axes[1,0].barh(np.arange(m), 1, color='C1', alpha=0.5)
axes[1,0].set(ylim=(-0.5,m-0.5))
axes[1,0].invert_xaxis()
axes[1,0].invert_yaxis()

axes[1,1].imshow(gamma.T, cmap='Greys', vmin=0)
axes[1,2].imshow(D.T, cmap='Greys', vmin=0)

axes[0,2].text(0.5,0.5, f"Uniform scipy.linprog EMD={emd}", ha='center')
fig.tight_layout()
fig.savefig("Plots/WassersteinUniform.svg")
plt.show()