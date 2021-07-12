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
from Datasets.EVS2020 import EuropeanValueStudy
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Wasserstein import EuclideanDistance, ConstraintMatrix

timer = Timer('EVS Exact Wasserstein')

EVS = EuropeanValueStudy(max_entries=100)
labels  = EVS.labeldict()
dataset = EVS.data

nuts = dataset.index.unique()
N = len(nuts)

K = np.zeros((N,N))

for i in range(N):
    for j in range(i+1, N):
        
        U = dataset.loc[nuts[i]]
        V = dataset.loc[nuts[j]]
        
        D = EuclideanDistance(U, V)
        n, m = len(U), len(V)
    
        # timer.add(f'Computed {n}x{m} distance matrix of {nuts[i]} and {nuts[j]}')
        
        A = ConstraintMatrix(n,m)
        b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
        c = D.reshape(-1)
        
        # timer.add(f'Created {n+m}x{n*m} constraint matrix')
        
        opt_res = linprog(-b, A.T, c, bounds=[None, None], method='highs')
        emd = -opt_res.fun
        
        K[i,j] = emd
        
    timer.add(f'Computed Wasserstein distance: {emd}')

K = K + K.T

np.save('Datasets/EVS2020/Precomputed Distances/small', K)  
timer.finish('Plots/.logfile.txt')
        