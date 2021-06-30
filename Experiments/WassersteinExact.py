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

dataset, labels = Data.LoadEVS(Data.small, 
                               countries=None,
                               transform=False, 
                               NUTS=1,
                               min_entries=40) 

A = dataset.loc['DE1'].values
B = dataset.loc['AL0'].values

D = np.sqrt(np.linalg.norm(A, ord=2, axis=1).reshape(-1,1)**2 +
            np.linalg.norm(B, ord=2, axis=1).reshape(1,-1)**2 -
            2 * np.inner(A,B))
