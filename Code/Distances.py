#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:10:01 2021

@author: fsvbach
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# d = cdist(Y1, Y2)
# assignment = linear_sum_assignment(d)
# print(d[assignment].sum() / n)

def GaussianWasserstein(N1, N2):
    m1, cov1 = N1.mean, N1.cov
    m2, cov2 = N2.mean, N2.cov
    tmp = cov2.sqrt() @ cov1.array() @ cov2.sqrt()
    s, P = np.linalg.eig(tmp)
    tmp = cov1.array() + cov2.array() - 2 * P @ np.diag(np.sqrt(s)) @ P.T 
    return np.linalg.norm(N1.mean - N2.mean)**2 + np.sum(np.diag(tmp))