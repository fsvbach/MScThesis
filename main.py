#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import time
start = time.perf_counter()

w_range    = [0, 0.25, 0.5, 0.75, 1]
w_range    = [0.5]
experiment = "TEST"
sklearn    = True

if sklearn:
    from sklearn.manifold import TSNE, MDS
else:
    from openTSNE import TSNE
    
from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGM, plotHGM2, plotTSNE, plotMDS


mixture = HierarchicalGaussianMixture(seed=13,
                                          datapoints=100, 
                                          samples=30, 
                                          features=2, 
                                          classes=3,
                                          ClassDistance=40,
                                          ClassVariance=80,
                                          DataVariance=5)

plotHGM(mixture, prefix=experiment)
plotHGM2(mixture, prefix=experiment)

for w in w_range:
    matrix = WassersteinDistanceMatrix(mixture.data_estimates(), w=w)
    # info   = (mixture.C, mixture.N, w, experiment)
    # plotTSNE(TSNE, matrix, info=info, sklearn=sklearn)
    # plotMDS(MDS, matrix, info=info)
    
stop = time.perf_counter()
print(f'Succesfully finished code in {stop-start} seconds.')