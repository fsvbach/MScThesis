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
              'ClassDistance': 25,
              'ClassVariance': 5}
         
    def __init__(self, seed=None, **kwargs):
        self.config.update(kwargs)
        
        self.N = self.config['datapoints']
        self.M = self.config['samples']
        self.F = self.config['features']
        self.K = self.config['classes']
        
        self.a = self.config['ClassDistance']
        self.b = self.config['ClassVariance']
        
        self.seed = seed
        self.generator = RandomGenerator(seed)
        
        self.set_params()
        self.data = self.generate_data()
                
    def _info(self):
        return f'''Random seed: {self.seed}, ClassDistance: {self.config['ClassDistance']}, ClassVariance: {self.config['ClassVariance']}\n{self.K} classes à {self.N} datapoints with each {self.M} samples in {self.F} dimensions'''
    
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
        
        dataset = []
        
        for Gaussian, Wishart in zip(self.ClassGaussians, self.ClassWisharts):

            data_means = self.generator.GaussianSamples(Gaussian, self.N)
            data_covs  = self.generator.WishartSamples(Wishart, self.N)
            
            for mean, cov in zip(data_means, data_covs):
                datapoint = GaussianDistribution(mean, cov)
                dataset.append(self.generator.GaussianSamples(datapoint, self.M))

        index = pd.MultiIndex.from_product([range(self.K), range(self.N), range(self.M)], names=["class", "datapoint", "sample"])
        dataset = pd.DataFrame(np.vstack(dataset), index = index)
        labels = dataset.indexA.index(0)
        
    
    def data_means(self):
        data = self.data.reshape((-1,self.D, self.F))
        return np.mean(data, axis=1)
    
    def data_covs(self):
        data = self.data.reshape((-1,self.D, self.F))
        data = (data.transpose([1,0,2]) - self.data_means()).transpose([1,0,2])
        return np.matmul(data.transpose([0,2,1]),data) / (self.D - 1)

    def data_estimates(self):
        means = self.data_means()
        covs  = self.data_covs()
        Gaussians = []
        for m,C in zip(means, covs):
            Gaussians.append(GaussianDistribution(m, CovarianceMatrix(C, from_array=True)))
        return Gaussians
    
    def labels(self):
        return np.hstack([np.ones(self.N,dtype=np.int) * i for i in range(self.C)]) 
    
        
