#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

# import time
# start = time.perf_counter()

experiment = "LARGE"
sklearn = True

if sklearn:
    from sklearn.manifold import TSNE
else:
    from openTSNE import TSNE
    
from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGM, plotHGM2, plotTSNE


mixture = HierarchicalGaussianMixture(seed=7,
                                          datapoints=50, 
                                          samples=20, 
                                          features=2, 
                                          classes=9,
                                          ClassDistance=25,
                                          ClassVariance=50,
                                          DataVariance=5)

plotHGM(mixture, prefix=experiment)
plotHGM2(mixture, prefix=experiment)

for w in [0, 0.25, 0.5, 0.75]:
    plotTSNE(TSNE, mixture, WassersteinDistanceMatrix, w=w, prefix=experiment, sklearn=sklearn)

# stop = time.perf_counter()
# print(f'Succesfully finished code in {stop-start} seconds.')