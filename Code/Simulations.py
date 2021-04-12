#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:34:16 2021

@author: fsvbach
"""

from scipy.stats import ortho_group
import numpy as np
generator = np.random.default_rng(seed=11)

# def GaussianMixture(means, covs, weights)

class CovarianceMatrix:
    def __init__(self, dim, params=None, maxvar=1):
        self.F = dim
        self.b = maxvar 
        
        if params:
            self.P, self.s = params
        else:
            self.P = ortho_group.rvs(dim=dim)
            self.s = generator.random(size=dim) * maxvar

    def array(self):
        return self.P@np.diag(self.s)@self.P.T
    
    def neighbours(self, size=1, noise=1):
        S = generator.multivariate_normal(mean = self.s, 
                                          cov  = np.eye(self.F) * noise, 
                                          size = size)
        return [CovarianceMatrix(dim=self.F, maxvar=self.b, params=(self.P,s)) for s in S]

    def contour(self, mean, grid):
        pass
    
    def samples(self, mean, size):
        pass
        
def uniform_means(dim, size=1, maxval=10):
    means = generator.random((size,dim)) - 0.5 
    return 2*means*maxval

def uniform_covs(dim, size=1, maxvar=1):
    return [CovarianceMatrix(dim, maxvar=maxvar) for i in range(size)]
    
class HierarchicalGaussian:
            
    config = {'maxClassVariance': 5,
              'maxDataVariance': 1,
              'DataNoise': 1}
         
    def __init__(self, datapoints=500, samples=20, features=2, classes=5, **kwargs):
        self.N = datapoints
        self.D = samples
        self.F = features
        self.C = classes

        self.config.update(kwargs)
        
        self.generate()
    
    def generate(self):   
        self.class_means = uniform_means(self.F, size=self.C)
        self.class_covs  = uniform_covs(self.F, size=self.C, maxvar=self.config['maxDataVariance'])
        
        classes = generator.integers(0, self.C, size=self.N)
        _ , counts = np.unique(classes, return_counts=True)
        
        self.data_means = []
        self.datapoints = []
        for cmean, ccov, count in zip(self.class_means, self.class_covs, counts):
            data_means = generator.multivariate_normal(mean= cmean, 
                                                       cov = np.eye(self.F) * self.config['maxClassVariance'], 
                                                       size= count)
            # data_covs  = ccov.neighbours(size=counts, noise=self.config['DataNoise'])
        
            for dmean in data_means:
                datapoints = generator.multivariate_normal(mean= dmean, 
                                                            cov = ccov.array(), 
                                                            size= self.D)
                self.datapoints.append(datapoints)
            
            self.data_means.append(data_means)
            
        self.data_means = np.vstack(self.data_means)
        self.datapoints = np.vstack(self.datapoints)

         
