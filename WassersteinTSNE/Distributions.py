#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:34:16 2021

@author: fsvbach
"""

from scipy.stats import special_ortho_group, wishart
from scipy.linalg import eigh
import numpy as np

class RandomGenerator:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)
    
    def UniformInteger(self, lower=0, upper=10000):
        return self.generator.integers(lower, upper)
    
    def UniformVector(self, dim=2, upper=1, size=1):
        samples = self.generator.random((size, dim)) - 0.5 
        return 2*samples*upper
    
    def OrthogonalMatrix(self, dim=2, size=1):
        return special_ortho_group.rvs(dim=dim, random_state=self.UniformInteger(), size=size)
    
    def UniformCovariance(self, dim=2, maxstd=1):
        P = self.OrthogonalMatrix(dim=dim)
        s = (self.generator.random(size=dim) * maxstd)**2
        return CovarianceMatrix(P,s)
        
    def GaussianSamples(self, Gaussian, size):
        return self.generator.multivariate_normal(mean = Gaussian.mean, 
                                                  cov  = Gaussian.cov.array(),
                                                  size = size)
    def WishartSamples(self, Wishart, size):
        return wishart.rvs(Wishart.nu, Wishart.scale, random_state=self.UniformInteger(), size=size)
  
                      
class WishartDistribution:
    def __init__(self, nu, scale):
        self.nu = nu
        self.scale  = scale
        
    def shape(self, std=1):
        return self.scale.shape(std=std)


class GaussianDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov  = cov
        
    def shape(self, std=1):
        width, height, angle = self.cov.shape(std=std)
        return self.mean, width, height, angle
    
class CovarianceMatrix:
    def __init__(self, P=None, s=None, from_array=False):
        '''
        stores eigen-decomposition of covariance matrix
        ----------
        P : np.ndarray
            orthogonal matrix.
        s : np.array
            eigenvalues in array.
        '''
        if from_array:
            s, P = eigh(P)
            s[np.where(s<0)]=0
            assert np.all(s>=0)
            
        self.P = P
        self.s = s
        
    def array(self):
        return self.P@np.diag(self.s)@self.P.T
    
    def sqrt(self):
        return self.P@np.diag(np.sqrt(self.s))@self.P.T
    
    def shape(self, std=1):
        assert len(self.s) == 2
        angle         = np.degrees(np.arctan2(*self.P[:,0][::-1]))
        width, height = np.sqrt(self.s)*2*std
        return width, height, angle