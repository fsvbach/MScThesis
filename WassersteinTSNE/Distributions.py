#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:34:16 2021

@author: fsvbach
"""

from scipy.stats import special_ortho_group, wishart
from scipy.linalg import eigh
import numpy as np

class CovarianceMatrix:
    def __init__(self, P=np.array([[3,4],[-4,3]])/5, s=np.array([1,3])):
        '''
        stores eigen-decomposition of covariance matrix
        ----------
        P : np.ndarray
            orthogonal matrix.
        s : np.array
            eigenvalues in array.
        '''       
        self.P = P
        self.s = s
        
    def diagonalize(self):
        self.s  = np.diag(self.array())
        self.P  = np.eye(len(self.s))
        
    def normalize(self):
        A = self.array()
        diagonal = np.sqrt(np.diag(A)) + 0.00000001
        A = (A/diagonal).T / diagonal
        self.s, self.P = eigh(A)
        self.s[np.where(self.s<0)]=0
            
    def array(self):
        return self.P@np.diag(self.s)@self.P.T
    
    def sqrt(self):
        return self.P@np.diag(np.sqrt(self.s))@self.P.T
    
    def shape(self, std=1):
        assert len(self.s) == 2
        angle         = np.degrees(np.arctan2(*self.P[:,0][::-1]))
        width, height = np.sqrt(self.s)*std*2
        return width, height, angle

def RotationMatrix(degree):
    return np.array([[np.cos(degree),-np.sin(degree)],
                     [np.sin(degree), np.cos(degree)]])    
    
def arr2cov(array):
    s, P = eigh(array)
    s[np.where(s<0)]=0
    assert np.all(s>=0)
    return CovarianceMatrix(P, s)   

class WishartDistribution:
    def __init__(self, nu=2, scale=CovarianceMatrix()):
        self.nu = nu
        self.scale  = scale
        
    def shape(self, std=1):
        return self.scale.shape(std=std)
        
class GaussianDistribution:
    def __init__(self, mean=np.array([1,0]), cov=CovarianceMatrix()):
        self.mean = mean
        self.cov  = cov
    
    def estimate(self, data):
        assert len(data) > 1
        self.mean = np.mean(data, axis=0)
        data     -= self.mean
        self.cov  = arr2cov(data.T @ data / (len(data) - 1))

    def shape(self, std=1):
        width, height, angle = self.cov.shape(std=std)
        return self.mean, width, height, angle

    def samples(self, size=20):
        return np.random.default_rng().multivariate_normal(mean = self.mean, 
                                                  cov  = self.cov.array(),
                                                  size = size)

class RandomGenerator:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)
    
    def RandomSeed(self):
        return self.UniformInteger(upper=10000)[0]
    
    def UniformInteger(self, lower=0, upper=10, size=1):
        return self.generator.integers(lower, upper, size)
    
    def UniformVector(self, dim=2, upper=1, size=1):
        samples = self.generator.random((size, dim)) - 0.5 
        return 2*samples*upper
    
    def OrthogonalMatrix(self, dim=2, size=1):
        return special_ortho_group.rvs(dim=dim, random_state=self.RandomSeed(), size=size)
    
    def UniformCovariance(self, dim=2, maxstd=1):
        P = self.OrthogonalMatrix(dim=dim)
        s = (self.generator.random(size=dim) * maxstd)**2
        return CovarianceMatrix(P,s)
        
    def GaussianSamples(self, Gaussian, size=1):
        return self.generator.multivariate_normal(mean = Gaussian.mean, 
                                                  cov  = Gaussian.cov.array(),
                                                  size = size)
    def WishartSamples(self, Wishart, size=1):
        arrays = wishart.rvs(Wishart.nu, Wishart.scale.array(), random_state=self.RandomSeed(), size=size)
        return [arr2cov(arr) for arr in arrays]

