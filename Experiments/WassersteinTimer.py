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

timer = Timer('Exact  Wasserstein')

EVS = EuropeanValueStudy(max_entries=2000)
labels  = EVS.labeldict()
dataset = EVS.data

A = dataset.loc['DK01'].values[:100]
B = dataset.loc['IS00'].values[:100]

D = EuclideanDistance(A, B)
n, m = len(A), len(B)

timer.add(f'Computed {n}x{m} distance matrix')

A = ConstraintMatrix(n,m)
b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
c = D.reshape(-1)

timer.add(f'Created {n+m}x{n*m} constraint matrix')

opt_res = linprog(-b, A.T, c, bounds=[None, None], method='highs')
emd = -opt_res.fun

timer.add(f'Computed Wasserstein distance: {emd}')
timer.finish('Plots/.logfile.txt')