#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:48:46 2021

@author: fsvbach
"""

from .HGM import HierarchicalGaussianMixture as HGM
from .TSNE import GaussianWassersteinDistance, WassersteinTSNE, Dataset2Gaussians
from .Timer import Timer
from .Distributions import *

class DataObject:
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels  = None
        
    def as_Gaussians(self, normalize=False, diagonal=False):
        return Dataset2Gaussians(self.dataset, diagonal=diagonal, normalize=normalize)
    
    def GWD(self):
        return GaussianWassersteinDistance(self.as_Gaussians())
    
    def TSNE(self, seed=None):
        return WassersteinTSNE(self.GWD(), seed=seed)