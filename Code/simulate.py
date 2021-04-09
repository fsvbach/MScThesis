#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:34:16 2021

@author: fsvbach
"""

from sklearn.datasets import make_spd_matrix
import numpy as np
generator = np.random.default_rng(seed=11)

class GaussianMixture:
    def __init__(self, datapoints=500, samples=20, features=2, classes=5):
        self.N = datapoints
        self.D = samples
        self.F = features
        self.C = classes
        
        self.generate()
    
    def generate(self):   
        self.class_means = (generator.random((self.C,self.F)) -0.5)*10
        
        classes = generator.integers(0, self.C, size=self.N)
        _ , counts = np.unique(classes, return_counts=True)
        data = []
        for mean, count in zip(self.class_means, counts):
            data.append(generator.multivariate_normal(mean= mean, 
                                                      cov = np.eye(self.F), 
                                                      size= count))
        self.data_means = np.vstack(data)
        
        
