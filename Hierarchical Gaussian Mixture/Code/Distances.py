#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:10:01 2021

@author: fsvbach
"""

import numpy as np

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
    
