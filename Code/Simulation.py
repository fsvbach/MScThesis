#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:34:16 2021

@author: fsvbach
"""

from scipy.stats import ortho_group
import numpy as np
generator = np.random.default_rng(seed=11)

class UniformDistribution:
    def __init__(self, dim, scale=1):
        self.scale = scale
        self.dim   = dim
    
    def samples(self, size=1):
        samples = generator.random((size,self.dim)) - 0.5 
        return 2*samples*self.scale
        
class GaussianDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov  = cov
    
    def samples(self, size=1):
        return generator.multivariate_normal(mean = self.mean, 
                                             cov  = self.cov.array(),
                                             size = size)
    def shape(self):
        return self.mean, np.sqrt(self.cov.s)*3, np.degrees(np.arctan2(*self.cov.P[:,0][::-1]))
        
class CovarianceMatrix:
    def __init__(self, dim, params=None, maxstd=1):
        self.F = dim
        self.maxstd = maxstd 
        
        if params:
            self.P, self.s = params
        else:
            self.P = ortho_group.rvs(dim=dim)
            self.s = (generator.random(size=dim) * maxstd)**2

    def array(self):
        return self.P@np.diag(self.s)@self.P.T
    
    def sqrt(self):
        return self.P@np.diag(np.sqrt(self.s))@self.P.T
    
    def neighbours(self, size=1, noise=1):
        S = generator.multivariate_normal(mean = self.s, 
                                          cov  = np.eye(self.F) * noise, 
                                          size = size)
        return [CovarianceMatrix(dim=self.F, maxstd=self.maxstd, params=(self.P,s)) for s in S]
    
    
class HierarchicalGaussian:
            
    config = {'ClassDistance': 25,
              'ClassVariance': 5,
              'DataVariance': 1}
         
    def __init__(self, datapoints=50, samples=20, features=2, classes=2, **kwargs):
        self.N = datapoints
        self.D = samples
        self.F = features
        self.C = classes

        self.config.update(kwargs)
 
        self.class_means = self.set_classes()
        self.data_means, self.samples  = self.set_datapoints()

    def set_classes(self):
        class_means = UniformDistribution(self.F, self.config['ClassDistance']).samples(size=self.C)
        # class_assmt = generator.integers(0, self.C, size=self.N)
        # _ , counts = np.unique(class_assmt, return_counts=True)
        
        self.classes = []      
        for cmean in class_means:
            ccov = CovarianceMatrix(self.F, maxstd=self.config['ClassVariance'])
            self.classes.append(GaussianDistribution(cmean, ccov))
        return class_means
            
    def set_datapoints(self):
        samples = []
        self.data  = []
        self.examples = []
        data_means = []
        
        for c in self.classes:
            means = c.samples(size=self.N)
            dcov = CovarianceMatrix(self.F, maxstd=self.config['DataVariance'])
            self.examples.append(GaussianDistribution(means[0], dcov))
            for dmean in means:
                datapoint = GaussianDistribution(dmean, dcov)
                self.data.append(datapoint)
                samples.append(datapoint.samples(self.D))
                
            data_means.append(means)
            
        return np.vstack(data_means), np.vstack(samples)
