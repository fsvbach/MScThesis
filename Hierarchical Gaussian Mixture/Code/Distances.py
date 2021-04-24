#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:10:01 2021

@author: fsvbach
"""

import numpy as np

def GaussianWasserstein(N1, N2, w=0.5):
    m1, cov1 = N1.mean, N1.cov
    m2, cov2 = N2.mean, N2.cov
    return (1-w)*np.linalg.norm(m1 - m2)**2 + (w)*np.linalg.norm(cov1.sqrt()-cov2.sqrt())**2
                                                               
def EuclideanDistanceMatrix(X):
    norms  = np.linalg.norm(X, axis=1, ord=2).reshape((len(X),1))**2
    matrix = norms + norms.T - 2 * X@X.T
    return matrix - matrix.min()

def WassersteinDistanceMatrix(X, w=0.5):
    N = len(X)
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            K[i,j] = GaussianWasserstein(X[i], X[j], w=w)
    return K + K.T