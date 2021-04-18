#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

# import time
# start = time.perf_counter()

from openTSNE import TSNE
from Code import Simulation, Distances
from Code.Visualization import plotHGM, plotTSNE

experiment = "TEST2"

mixture = Simulation.HierarchicalGaussian(seed=11,
                                          datapoints=50, 
                                          samples=20, 
                                          features=2, 
                                          classes=3,
                                          ClassDistance=50,
                                          ClassVariance=50,
                                          DataVariance=5)

euclid = Distances.EuclideanDistanceMatrix(mixture.data_means())
wasser = Distances.WassersteinDistanceMatrix(mixture.datapoints)

plotHGM(mixture, prefix=experiment)
plotTSNE(TSNE, euclid, name='Euclidean', prefix=experiment)
plotTSNE(TSNE, wasser, name='Wasserstein', prefix=experiment)

# stop = time.perf_counter()
# print(f'Succesfully finished code in {stop-start} seconds.')