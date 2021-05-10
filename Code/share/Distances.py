#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:48:46 2021

@author: fsvbach
"""

import numpy as np
from openTSNE import TSNE as openTSNE
from sklearn.manifold import TSNE as skleTSNE

class WassersteinTSNE:
    def __init__(self, seed=None, sklearn=False, load=None, store=None):
        self.sklearn = sklearn
        self.seed    = seed
        self.path    = load
        self.store   = store
        
    def fit(self, data):
        if self.path:
            return np.load(self.path)
        
        if self.sklearn:
            tsne = skleTSNE(random_state=self.seed)
            embedding = tsne.fit_transform(data)
        else:
            tsne = openTSNE(random_state=self.seed)
            embedding = tsne.fit(data)
    
        if self.store:
            np.save(self.store, embedding)
        
        return embedding


class GaussianWassersteinDistance:
    def __init__(self, Gaussians):
        means = []
        covs  = []
        for G in Gaussians:
            covs.append(G.cov.sqrt())
            means.append(G.mean)
        self.EDM = self.EuclideanDistanceMatrix(np.stack(means))
        self.CDM = self.CovarianceDistanceMatrix(np.stack(covs))
    
    def EuclideanDistanceMatrix(self, X):
        norms  = np.linalg.norm(X, axis=1, ord=2).reshape((len(X),1))**2
        matrix = norms + norms.T - 2 * X@X.T
        return matrix - matrix.min()
    
    def CovarianceDistanceMatrix(self, X):
        norms = np.linalg.norm(X, ord='fro', axis=(1, 2)).reshape((len(X),1))**2
        matrix = norms + norms.T - 2 * np.tensordot(X,X, axes=([1,2],[1,2]))
        return matrix - matrix.min()
    
    def matrix(self, w=0.5):
        return (1-w)*self.EDM + w*self.CDM
    
    
    
def PairwiseGaussianWasserstein(N1, N2, w=0.5):
    m1, cov1 = N1.mean, N1.cov
    m2, cov2 = N2.mean, N2.cov
    tmp = cov2.sqrt() @ cov1.array() @ cov2.sqrt()
    s, P = np.linalg.eig(tmp)
    tmp = cov1.array() + cov2.array() - 2 * P @ np.diag(np.sqrt(s)) @ P.T 
    return (1-w)*np.linalg.norm(m1 - m2)**2 + w*np.sum(np.diag(tmp))
                                               
def WassersteinMatrixLoop(X, w=0.5):
    N = len(X)
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            K[i,j] = PairwiseGaussianWasserstein(X[i], X[j], w=w)
    return K + K.T