#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:35:42 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np

from openTSNE import TSNE as openTSNE
from sklearn.manifold import TSNE as skleTSNE

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from .Distributions import GaussianDistribution, arr2cov


def accuracy(embedding, k=10):
    kNN    = KNeighborsClassifier(k)
    kNN.fit(embedding.values, embedding.index)
    test = kNN.predict(embedding.values)
    return accuracy_score(test, embedding.index)


def Dataset2Gaussians(dataset, diagonal=False, normalize=False):
    Gaussians = []
    names    = []
    for name, data in dataset.groupby(level=0):
        G = GaussianDistribution()
        G.estimate(data.values)
        if diagonal:
            G.cov.diagonalize()
        elif normalize:
            G.cov.normalize()
        names.append(name)
        Gaussians.append(G)
    return pd.Series(Gaussians, index=names)

class WassersteinTSNE:
    def __init__(self, GWD, seed=None, sklearn=False):
        self.GWD     = GWD
        self.sklearn = sklearn
        self.seed    = seed

    def fit(self, w):
        if self.sklearn:
            tsne = skleTSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=self.seed)
            embedding = tsne.fit_transform(self.GWD.matrix(w=w))
        else:
            tsne = openTSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=self.seed)
            embedding = tsne.fit(self.GWD.matrix(w=w))
        
        # cols = [(f'w={w}','x'), (f'w={w}', 'y')]
        embedding =  pd.DataFrame(embedding, 
                     index=self.GWD.index,
                     columns = ['x','y'])
        return embedding

            
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
    
    
class NormalTSNE:
    def __init__(self, seed=None, sklearn=False):
        self.sklearn = sklearn
        self.seed    = seed
        
    def fit(self, dataset):
        if self.sklearn:
            tsne = skleTSNE(random_state=self.seed)
            embedding = tsne.fit_transform(dataset.values)
        else:
            tsne = openTSNE(random_state=self.seed)
            embedding = tsne.fit(dataset.values)
            
        return  pd.DataFrame(embedding,
                             index=dataset.index,
                             columns = ['x','y']) 

