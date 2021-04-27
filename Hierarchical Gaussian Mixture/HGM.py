#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

from Code.Visualization import plotHGM
from Code.Simulation import HierarchicalGaussianMixture

mixture = HierarchicalGaussianMixture(seed=11,
                                    datapoints=40, 
                                    samples=20, 
                                    features=2, 
                                    classes=4,
                                    ClassDistance=10,
                                    ClassVariance=5,
                                    DataVariance=1)


plotHGM(mixture, 'CleanExample')