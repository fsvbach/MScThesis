#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

from .Visualization import HGM 
from DataLoader.Simulation import HierarchicalGaussianMixture

def run():
    mixture = HierarchicalGaussianMixture(seed=1,
                                        datapoints=300, 
                                        samples=5, 
                                        features=2, 
                                        classes=6,
                                        DataVariance=5)
    


    HGM.plotDataset(mixture, std=3, prefix='cleanExample')

    
