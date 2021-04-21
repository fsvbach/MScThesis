#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:34:16 2021

@author: fsvbach
"""

from scipy.stats import ortho_group
import numpy as np

class Generator:
    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed=seed)
    
    def UniformVector(self, dim, size, limit):
        samples = self.generator.random((size, dim)) - 0.5 
        return 2*samples*limit
    
    def UniformCovariance(self, dim, maxstd):
        seed = self.generator.integers(10000)
        P = ortho_group.rvs(dim=dim, random_state=seed)
        s = (self.generator.random(size=dim) * maxstd)**2
        return CovarianceMatrix(P,s)
        
    def GaussianSamples(self, Gaussian, size):
        return self.generator.multivariate_normal(mean = Gaussian.mean, 
                                                  cov  = Gaussian.cov.array(),
                                                  size = size)
        
class GaussianDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov  = cov
    
    def shape(self):
        angle         = np.degrees(np.arctan2(*self.cov.P[:,0][::-1]))
        width, height = np.sqrt(self.cov.s)*4
        return self.mean, width, height, angle


class CovarianceMatrix:
    def __init__(self, P, s):
        self.P = P
        self.s = s
        
    def array(self):
        return self.P@np.diag(self.s)@self.P.T
    
    def sqrt(self):
        return self.P@np.diag(np.sqrt(self.s))@self.P.T
    
    
class HierarchicalGaussianMixture:
            
    config = {'ClassDistance': 25,
              'ClassVariance': 5,
              'DataVariance': 1}
         
    def __init__(self, seed=None, datapoints=50, samples=20, features=2, classes=3, **kwargs):
        self.N = datapoints
        self.D = samples
        self.F = features
        self.C = classes
        
        self.config.update(kwargs)
        
        self.seed = seed
        self.generator = Generator(seed)
        
        self.data = self.generate_data()

    def _info(self):
        return f'''{self.C} classes à {self.N} datapoints with each {self.D} samples in {self.F} dimensions.\nRandom seed: {self.seed}, ClassDistance: {self.config['ClassDistance']}, ClassVariance: {self.config['ClassVariance']}, DataVariance: {self.config['DataVariance']}'''
        
    def generate_data(self):
        class_means = self.generator.UniformVector(self.F, self.C, self.config['ClassDistance'])
        
        cdata = []
        self.datapoints = []
        self.classes    = []   
        
        for cmean in class_means:
            ccov = self.generator.UniformCovariance(self.F, self.config['ClassVariance'])
            GaussianClass = GaussianDistribution(cmean, ccov)
            self.classes.append(GaussianClass)

            ddata = []
            data_means = self.generator.GaussianSamples(GaussianClass, self.N)
            dcov       = self.generator.UniformCovariance(self.F, self.config['DataVariance'])
            
            for dmean in data_means:
                datapoint = GaussianDistribution(dmean, dcov)
                samples   = self.generator.GaussianSamples(datapoint, self.D)
                
                self.datapoints.append(datapoint)
                ddata.append(samples)
                
            cdata.append(np.array(ddata))
        
        return np.array(cdata)
    
    def data_means(self):
        data = self.data.reshape((-1,self.D, self.F))
        return np.mean(data, axis=1)
    
    def data_covs(self):
        data = self.data.reshape((-1,self.D, self.F))
        data = (data.transpose([1,0,2]) - self.data_means()).transpose([1,0,2])
        return np.matmul(data.transpose([0,2,1]),data)

    def data_estimates(self):
        means = self.data_means()
        covs  = self.data_covs()
        Gaussians = []
        for m,C in zip(means, covs):
            s, P = np.linalg.eig(C)
            Gaussians.append(GaussianDistribution(m, CovarianceMatrix(P, s)))
        return Gaussians
    
        