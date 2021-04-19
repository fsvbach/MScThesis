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

def AlternativeGaussianWasserstein(N1, N2, w=0.5):
    m1, cov1 = N1.mean, N1.cov
    m2, cov2 = N2.mean, N2.cov
    tmp = cov2.sqrt() @ cov1.array() @ cov2.sqrt()
    s, P = np.linalg.eig(tmp)
    tmp = cov1.array() + cov2.array() - 2 * P @ np.diag(np.sqrt(s)) @ P.T 
    return (1-w)*np.linalg.norm(m1 - m2)**2 + (w)*np.sum(np.diag(tmp))

def GaussianWasserstein(N1, N2, w=0.5):
    m1, cov1 = N1.mean, N1.cov
    m2, cov2 = N2.mean, N2.cov
    return (1-w)*np.linalg.norm(m1 - m2)**2 + (w)*np.linalg.norm(cov1.sqrt()-cov2.sqrt())**2
                                                               
def EuclideanDistanceMatrix(X):
    norms  = np.linalg.norm(X, axis=1, ord=2).reshape((len(X),1))**2
    matrix = norms + norms.T - 2 * X@X.T
    return matrix - matrix.min()

def WassersteinDistanceMatrix(mixture, w=0.5):
    X = mixture.datapoints
    N = mixture.N * mixture.C
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            K[i,j] = GaussianWasserstein(X[i], X[j], w=w)
    return K + K.T