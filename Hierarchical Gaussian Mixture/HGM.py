#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

from Code.Visualization import plotHGM
from Code.Simulation import HierarchicalGaussianMixture

mixture = HierarchicalGaussianMixture(seed=1,
                                    datapoints=300, 
                                    samples=5, 
                                    features=2, 
                                    classes=6,
                                    DataVariance=5)



plotHGM(mixture, std=3)

    
