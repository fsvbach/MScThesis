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
from WassersteinTSNE.Distances import EuclideanDistance, ConstraintMatrix, SparseConstraint

timer = Timer('Exact  Wasserstein')

EVS = EuropeanValueStudy(max_entries=2000)
labels  = EVS.labeldict()
dataset = EVS.data

A = dataset.loc['BABIH'].values
B = dataset.loc['IS00'].values

D = EuclideanDistance(A, B)
n, m = len(A), len(B)

timer.add(f'Computed {n}x{m} distance matrix')

b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
c = D.reshape(-1)

try:
    A = ConstraintMatrix(n,m)
    timer.add(f'Created {n+m}x{n*m} constraint matrix')
except:
    print('Cannot allocate disk space for constraint')

try:
    opt_res = linprog(-b, A.T, c, bounds=[None, None], options={"tol": 1e-4}, method='highs')
    emd = -opt_res.fun
    timer.add(f'Computed dense Wasserstein distance: {emd}')
except:
    print("Cannot solve linprog")

A = SparseConstraint(n,m)

timer.add(f'Created sparse constraint matrix')

opt_res = linprog(-b, A.T, c, bounds=[None, None], method='highs', options={"sparse":True,"tol": 1e-4})
emd = -opt_res.fun

timer.add(f'Computed sparse Wasserstein distance: {emd}')
timer.finish('Plots/.logfile.txt')