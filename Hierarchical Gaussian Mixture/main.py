#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code.utils import Timer

w_range    = [0, 0.25, 0.5, 0.75, 1]
# w_range    = [0.5]
seed       = 13
experiment = "LOW"
sklearn    = True
timer      = Timer(experiment, output=True)

if sklearn:
    from sklearn.manifold import TSNE, MDS
else:
    from openTSNE import TSNE
    
from Code.Distances import WassersteinDistanceMatrix
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGMdata, plotHGMclasses, plotTSNE, plotMDS, plotAccuracy

mixture = HierarchicalGaussianMixture(seed=seed,
                                    datapoints=300, 
                                    samples=30, 
                                    features=2, 
                                    classes=7,
                                    ClassDistance=4,
                                    ClassVariance=6,
                                    DataVariance=5)

if mixture.F == 2:
    plotHGMdata(mixture, prefix=experiment, std=3)
    plotHGMclasses(mixture, prefix=experiment, std=1, n=1)

timer.add(f'{mixture._info()}\n\nCreated data')

accuracies = plotAccuracy(labels=mixture.labels(), 
                          params=w_range, 
                          prefix=experiment, 
                          k=10)

for w in w_range:
    info   = (mixture.C, mixture.N, w, seed, experiment)
    
    matrix = WassersteinDistanceMatrix(mixture.data_estimates(), w=w)
    timer.add(f'Computed matrix with w={w}')
    
    embedding = plotTSNE(TSNE, matrix, info=info, sklearn=sklearn)
    timer.add('Done TSNE')
    
    acc = accuracies.append(embedding)
    timer.add(f'Accuracy: {acc}%')
    # plotMDS(MDS, matrix, info=info)
    # timer.add('Done MDS')

accuracies.plot()

timer.finish()