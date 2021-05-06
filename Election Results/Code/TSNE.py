#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:48:46 2021

@author: fsvbach
"""

import numpy as np
from openTSNE import TSNE as openTSNE
from sklearn.manifold import TSNE as skleTSNE

class myTSNE:
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
