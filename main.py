#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

# import time
# start = time.perf_counter()

experiment = "MEDIUM"
sklearn = True

if sklearn:
    from sklearn.manifold import TSNE
else:
    from openTSNE import TSNE
    
from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGM, plotTSNE


mixture = HierarchicalGaussianMixture(seed=15,
                                          datapoints=50, 
                                          samples=30, 
                                          features=2, 
                                          classes=4,
                                          ClassDistance=1,
                                          ClassVariance=1,
                                          DataVariance=1)

plotHGM(mixture, prefix=experiment)

for w in [0.25, 0.5, 0.75, 1]:
    plotTSNE(TSNE, mixture, WassersteinDistanceMatrix, w=w, prefix=experiment, sklearn=sklearn)

# stop = time.perf_counter()
# print(f'Succesfully finished code in {stop-start} seconds.')