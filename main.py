#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

# import time
# start = time.perf_counter()

from openTSNE import TSNE
from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGM, plotTSNE

experiment = "MEDIUM"

mixture = HierarchicalGaussianMixture(seed=11,
                                          datapoints=50, 
                                          samples=20, 
                                          features=2, 
                                          classes=4,
                                          ClassDistance=25,
                                          ClassVariance=25,
                                          DataVariance=25)

plotHGM(mixture, prefix=experiment)

import numpy as np
for w in np.linspace(0,1,4):
    plotTSNE(TSNE, mixture, metric=WassersteinDistanceMatrix, w=np.round(w,2), prefix=experiment)

# stop = time.perf_counter()
# print(f'Succesfully finished code in {stop-start} seconds.')