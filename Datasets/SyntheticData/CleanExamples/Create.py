#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:19:11 2021

@author: bachmafy
"""

import numpy as np

from WassersteinTSNE import HierarchicalGaussianMixture as HGM 
from WassersteinTSNE import RotationMatrix, GaussianDistribution, WishartDistribution, CovarianceMatrix

path = 'Datasets/SyntheticData/CleanExamples'

def label(mixture):
    dataset = mixture.data
    dataset['ClassLabels'] = dataset.index.get_level_values(0).map(mixture.labeldict())
    dataset.set_index('ClassLabels', append=True, inplace=True)
    return dataset.droplevel(1).swaplevel()

def Distinct():
    mixture = HGM(  seed=13,
                    datapoints=100, 
                    samples=30, 
                    features=2, 
                    classes=4,
                    random=True)
    
    A = CovarianceMatrix(RotationMatrix(0), s=[20,1])
    B = CovarianceMatrix(RotationMatrix(90), s=[20,1])
    C = CovarianceMatrix(RotationMatrix(0), s=[20,1])
    D = CovarianceMatrix(RotationMatrix(90), s=[20,1])
    
    mixture.set_params(means   = np.array([[40,0],[40,0],[0,0],[0,0]]),
                       Gammas = [CovarianceMatrix(s=[5,5])]*4,
                       nus     = np.ones(4)*5,
                       Lambdas  = [A,B,C,D])  
    
    dataset = label(mixture)
    dataset.to_csv(f'{path}/Distinct.csv')
    return mixture

def Random():
    mixture = HGM(  seed=11,
                    datapoints=40, 
                    samples=20, 
                    features=2, 
                    classes=4,
                    ClassMeanDistance=10,
                    ClassScaleVariance=5)
    
    dataset = label(mixture)
    dataset.to_csv(f'{path}/Random.csv')
    return mixture


def DoubleCross():
    mixture = HGM(  seed=13,
                    datapoints=100, 
                    samples=30, 
                    features=2, 
                    classes=4,
                    random=False)
    
    C = CovarianceMatrix(RotationMatrix(0), s=[20,1])
    D = CovarianceMatrix(RotationMatrix(90), s=[20,1])
    
    mixture.set_params(means   = np.array([[25,0],[25,0],[0,0],[0,0]]),
                       Gammas = [CovarianceMatrix(s=[5,5])]*4,
                       nus     = np.ones(4)*4,
                       Lambdas  = [C,D,C,D])
    
    mixture.generate_data()
    dataset = label(mixture)
    dataset.to_csv(f'{path}/DoubleCross.csv')
    return mixture


