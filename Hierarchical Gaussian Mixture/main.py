#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code.utils import Timer

w_range    = [0, 0.25, 0.5, 0.75, 1]
# w_range    = [0.5]
experiment = "FEWpoints"
seed       = 13
sklearn    = True
timer      = Timer(experiment)

if sklearn:
    from sklearn.manifold import TSNE, MDS
else:
    from openTSNE import TSNE
    
from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGM, plotHGM2, plotTSNE, plotMDS

mixture = HierarchicalGaussianMixture(seed=seed,
                                    datapoints=100, 
                                    samples=5, 
                                    features=2, 
                                    classes=5,
                                    ClassDistance=40,
                                    ClassVariance=80,
                                    DataVariance=5)

if mixture.F == 2:
    plotHGM(mixture, prefix=experiment)
    plotHGM2(mixture, prefix=experiment)

timer.add(f'{mixture._info()}\nCreated data')

for w in w_range:
    info   = (mixture.C, mixture.N, w, experiment)
    
    matrix = WassersteinDistanceMatrix(mixture.data_estimates(), w=w)
    timer.add(f'Computed matrix with w={w}')
    
    plotTSNE(TSNE, matrix, info=info, sklearn=sklearn)
    timer.add('Done TSNE')
    
    # plotMDS(MDS, matrix, info=info)
    # timer.add('Done MDS')

timer.finish()