#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""


from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGM, plotHGM2, plotTSNE, plotMDS

w_range    = [0, 0.25, 0.5, 0.75, 1]
# w_range    = [0.5]
experiment = "HIGHdim"
seed       = 13
sklearn    = True

if sklearn:
    from sklearn.manifold import TSNE, MDS
else:
    from openTSNE import TSNE
    
mixture = HierarchicalGaussianMixture(seed=seed,
                                    datapoints=100, 
                                    samples=30, 
                                    features=20, 
                                    classes=9,
                                    ClassDistance=40,
                                    ClassVariance=80,
                                    DataVariance=5)

if mixture.F == 2:
    plotHGM(mixture, prefix=experiment)
    plotHGM2(mixture, prefix=experiment)

print(f'{mixture._info()}\nCreated data in {time.perf_counter()-start} seconds.')

for w in w_range:
    info   = (mixture.C, mixture.N, w, experiment)
    
    matrix = WassersteinDistanceMatrix(mixture.data_estimates(), w=w)
    print(f'Computed matrix with w={w} in {time.perf_counter()-start} seconds.')
    
    plotTSNE(TSNE, matrix, info=info, sklearn=sklearn)
    print(f'Done TSNE in { time.perf_counter()-start} seconds.')
    
    plotMDS(MDS, matrix, info=info)
    print(f'Done MDS in { time.perf_counter()-start} seconds.')

print(f'Succesfully finished code in {time.perf_counter()-start} seconds.')