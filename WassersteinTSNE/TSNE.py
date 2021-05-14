#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:35:42 2021

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

    def fit_precomputed(self, matrix):
        
        if self.sklearn:
            tsne = skleTSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=self.seed)
            embedding = tsne.fit_transform(matrix)
        else:
            tsne = openTSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=self.seed)
            embedding = tsne.fit(matrix)
            
        return embedding
            
            
class GaussianWassersteinDistance:
    def __init__(self, Gaussians, fast_approx=False):
        means = []
        covs  = []
        sqrts = []
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
        # print(tmp)
        s, P = np.linalg.eig(tmp) ##eigh 
        s[np.where(s<0)]=0
        tmp = cov1.array() + cov2.array() - 2 * P @ np.diag(np.sqrt(s)) @ P.T 
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
    
    
    

