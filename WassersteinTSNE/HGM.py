#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:36:56 2021

@author: fsvbach
"""
import numpy as np
import pandas as pd

from .Distributions import RandomGenerator, GaussianDistribution, WishartDistribution, CovarianceMatrix

class HierarchicalGaussianMixture:
            
    config = {'datapoints':60, 
              'samples' :20, 
              'features':2,
              'classes':4,
              'ClassMeanDistance': 25,
              'ClassScaleVariance': 5}
         
    def __init__(self, seed=None, random=True, **kwargs):
        self.config.update(kwargs)
        
        self.N = self.config['datapoints']
        self.M = self.config['samples']
        self.F = self.config['features']
        self.K = self.config['classes']
        
        self.a = self.config['ClassMeanDistance']
        self.b = self.config['ClassScaleVariance']
        
        self.seed = seed
        self.generator = RandomGenerator(seed)
                
        if random:
            self.info = self._info()
            self.set_params()
            self.generate_data()
        else:
            self.info = f'Custom parameters with Random seed {self.seed}'
            self.ClassGaussians = None
            self.ClassWisharts  = None
                    
    def _info(self):
        return f'''Random seed: {self.seed}, ClassMeanDistance: {self.a}, ClassScaleVariance: {self.b}\n{self.K} classes à {self.N} datapoints with each {self.M} samples in {self.F} dimensions'''
    
    def set_params(self, means=None, Lambdas=None, nus=None, Gammas=None):
        prior = lambda b: WishartDistribution(self.F, CovarianceMatrix(np.eye(self.F), b*np.ones(self.F)))
        
        if not means:
            means = self.generator.UniformVector(self.F, self.a, self.K)
        assert means.shape == (self.K, self.F)
        
        if not nus:
            nus = self.generator.UniformInteger(lower=self.F, upper=self.F*2, size=self.K)
        assert nus.shape == (self.K,)
        
        if not Gammas:
            Gammas = self.generator.WishartSamples(prior(self.b), self.K)
        assert Gammas[0].array().shape == (self.F, self.F)

        if not Lambdas:
            Lambdas = self.generator.WishartSamples(prior(1), self.K)
        assert Lambdas[0].array().shape == (self.F, self.F)
        
        self.ClassGaussians = [GaussianDistribution(mean, Cov) for mean, Cov in zip(means, Gammas)]
        self.ClassWisharts  = [WishartDistribution(nu, Scale) for nu, Scale in zip(nus, Lambdas)]           
        
    def generate_data(self): 
        assert self.ClassGaussians and self.ClassWisharts  
        
        dataset = []
        
        for Gaussian, Wishart in zip(self.ClassGaussians, self.ClassWisharts):

            data_means = self.generator.GaussianSamples(Gaussian, self.N)
            data_covs  = self.generator.WishartSamples(Wishart, self.N)
            
            for mean, cov in zip(data_means, data_covs):
                datapoint = GaussianDistribution(mean, cov)
                dataset.append(self.generator.GaussianSamples(datapoint, self.M))

        index       = pd.MultiIndex.from_product([range(self.K*self.N), range(self.M)], names=["datapoint", "sample"])
        self.data   = pd.DataFrame(np.vstack(dataset), index = index)
        self.labels = self.labels()
        return self.data
    
    def labels(self):
        labels = np.hstack([np.ones(self.N,dtype=np.int) * i for i in range(self.K)])
        return pd.Series(labels, dtype=np.int, name='ClassLabel' )
    
