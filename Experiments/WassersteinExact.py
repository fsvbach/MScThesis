#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:50:27 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.stats import wasserstein_distance
from Datasets import EVS2020 as Data

def EuclideanDistance(A,B):
    N1 = np.linalg.norm(A, ord=2, axis=1).reshape(-1,1)**2 
    N2 = np.linalg.norm(B, ord=2, axis=1).reshape(1,-1)**2 
    N3 = -2 * np.inner(A,B)
    D  = N1 + N2 + N3
    D[np.where(D<0)] = 0
    return np.sqrt(D)
    
def ConstraintMatrix(n,m):
    N = np.repeat(np.identity(n), m, axis=1)
    M = np.hstack([np.identity(m)]*n)
    return np.vstack([N,M])
    
dataset, labels = Data.LoadEVS(Data.small, 
                               countries=None,
                               transform=False, 
                               NUTS=1,
                               min_entries=40) 

A = dataset.loc['DE1'].values[:200]
B = dataset.loc['AL0'].values[:50]

D = EuclideanDistance(A, B)
plt.imshow(D)
plt.show()

n, m = len(A), len(B)

A = ConstraintMatrix(n,m)
b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
c = D.reshape(-1)

opt_res = linprog(-b, A.T, c, bounds=[None, None])
emd = -opt_res.fun