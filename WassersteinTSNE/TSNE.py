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

from .Distributions import RotationMatrix

class GaussianTSNE:
    def __init__(self, GWD, seed=None, sklearn=False):
        self.GWD     = GWD
        self.sklearn = sklearn
        self.seed    = seed

    def fit(self, w, trafo=None):
        if self.sklearn:
            tsne = skleTSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=self.seed)
            embedding = tsne.fit_transform(self.GWD.matrix(w=w)**2)
        else:
            tsne = openTSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=self.seed)
            embedding = tsne.fit(self.GWD.matrix(w=w)**2)
        
        if trafo is None:
            trafo = np.eye(2)
        embedding =  pd.DataFrame( embedding @ trafo, 
                     index=self.GWD.index,
                     columns = ['x','y'])
        return embedding
    
    def accuracy(self, w, labeldict, k=10):
        embedding = self.fit(w)
        embedding.index = embedding.index.to_series().map(labeldict)
        kNN    = KNeighborsClassifier(k)
        kNN.fit(embedding.values, embedding.index)
        test = kNN.predict(embedding.values)
        return accuracy_score(test, embedding.index)

            
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

class WassersteinTSNE:
    def __init__(self, seed=None, sklearn=False):
        self.sklearn = sklearn
        self.seed    = seed
        
    def fit(self, dataset, trafo=None):
        if self.sklearn:
            tsne = skleTSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=self.seed)
            embedding = tsne.fit_transform(dataset.values**2)
        else:
            tsne = openTSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=self.seed)
            embedding = tsne.fit(dataset.values**2)
        
        if trafo is None:
            trafo = np.eye(2)
        embedding =  pd.DataFrame(embedding @ trafo, 
                     index=dataset.index,
                     columns = ['x','y'])
        return embedding