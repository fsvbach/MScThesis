#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:41:49 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from Datasets import EVS2020 as Data

dataset, labels = Data.LoadEVS(Data.small, 
                               countries=None,
                               transform=False, 
                               NUTS=1,
                               min_entries=40) 

P1 = np.histogramdd(dataset.loc['DE1', ('v102', 'v38')].values, bins=10, density=False)
P2 = np.histogramdd(dataset.loc['AL0', ('v102', 'v38')].values, bins=10, density=False)

P1 = P1[0]/P1[0].sum()
P2 = P2[0]/P2[0].sum()

### We construct our A matrix by creating two 3-way tensors,
### and then reshaping and concatenating them

l=10
D = np.ndarray(shape=(l, l))

for i in range(l):
 	for j in range(l):
		D[i,j] = abs(range(l)[i] - range(l)[j])
        
A_r = np.zeros((l, l, l))
A_t = np.zeros((l, l, l))

for i in range(l):
 	for j in range(l):
		A_r[i, i, j] = 1
		A_t[i, j, i] = 1

A = np.concatenate((A_r.reshape((l, l**2)), A_t.reshape((l, l**2))), axis=0)
b = np.concatenate((P1, P2), axis=0)
c = D.reshape((l**2))

opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None])
emd = opt_res.fun
gamma = opt_res.x.reshape((l, l))

fig.savefig('Plots/WassersteinExperiments1D.svg')
plt.show()
plt.close()