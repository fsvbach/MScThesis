#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:36:56 2021

@author: fsvbach
"""
import numpy as np
import pandas as pd

from .Distributions import RandomGenerator, GaussianDistribution, CovarianceMatrix

class HierarchicalGaussianMixture:
            
    config = {'datapoints':50, 
              'samples' :20, 
              'features':2,
              'classes':3,
              'ClassDistance': 25,
              'ClassVariance': 5,
              'DataVariance': 1}
         
    def __init__(self, seed=None, **kwargs):
        self.config.update(kwargs)
        
        self.N = self.config['datapoints']
        self.D = self.config['samples']
        self.F = self.config['features']
        self.C = self.config['classes']
        
        self.seed = seed
        self.generator = RandomGenerator(seed)
        
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
    
        
