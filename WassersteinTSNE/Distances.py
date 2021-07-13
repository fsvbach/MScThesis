#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:57:05 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np

from scipy.optimize import linprog
from .Distributions import arr2cov

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

def WassersteinDistanceMatrix(dataset):
    data = dataset.index.unique()
    N = len(data)
    K = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i+1, N):
            U = dataset.loc[data[i]]
            V = dataset.loc[data[j]]

            D = EuclideanDistance(U, V)
            n, m = len(U), len(V)

            A = ConstraintMatrix(n,m)
            b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
            c = D.reshape(-1)

            opt_res = linprog(-b, A.T, c, bounds=[None, None], method='highs')      
            K[i,j] = -opt_res.fun
        
    return K + K.T

class GaussianWassersteinDistance:
    def __init__(self, Gaussians, fast_approx=False):
        sqrts = []
        means = []
        covs  = []
        
        self.index = Gaussians.index
        for G in Gaussians:            
            sqrts.append(G.cov.sqrt())
            covs.append(G.cov)
            means.append(G.mean)
            
        self.EDM = self.EuclideanDistanceMatrix(np.stack(means))
        
        if fast_approx:
            self.CDM = self.FrobeniusDistanceMatrix(np.stack(sqrts))
        else:
            self.CDM = self.CovarianceDistanceLoop(covs)
         
    def EuclideanDistanceMatrix(self, X):
        norms  = np.linalg.norm(X, axis=1, ord=2).reshape((len(X),1))**2
        matrix = norms + norms.T - 2 * X@X.T
        return matrix - matrix.min()
    
    def FrobeniusDistanceMatrix(self, X):
        norms = np.linalg.norm(X, ord='fro', axis=(1, 2)).reshape((len(X),1))**2
        matrix = norms + norms.T - 2 * np.tensordot(X,X, axes=([1,2],[1,2]))
        return matrix - matrix.min()
    
    def PairwiseCovarianceDistance(self, cov1, cov2):
        tmp = cov2.sqrt() @ cov1.array() @ cov2.sqrt()
        tmp = arr2cov(tmp)
        tmp = cov1.array() + cov2.array() - 2 * tmp.sqrt()
        return np.sum(np.diag(tmp))

    def CovarianceDistanceLoop(self, covs):
        N = len(covs)
        K = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                K[i,j] = self.PairwiseCovarianceDistance(covs[i], covs[j])
        return K + K.T
    
    def matrix(self, w=0.5):
        return (1-w)*self.EDM + w*self.CDM
    
    