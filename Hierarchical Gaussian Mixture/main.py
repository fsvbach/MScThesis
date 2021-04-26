#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code.utils import Timer

n_plots    = 11
seed       = 13
experiment = "FEW"
sklearn    = True
timer      = Timer(experiment, output=True)

if sklearn:
    from sklearn.manifold import TSNE, MDS
else:
    from openTSNE import TSNE
    
from Code.Distances import GaussianWassersteinDistance
from Code.Simulation import HierarchicalGaussianMixture
from Code.Visualization import plotHGMdata, plotHGMclasses, plotTSNE, plotMDS, plotAccuracy

mixture = HierarchicalGaussianMixture(seed=seed,
                                    datapoints=400, 
                                    samples=5, 
                                    features=2, 
                                    classes=6,
                                    ClassDistance=4,
                                    ClassVariance=5,
                                    DataVariance=6)

if mixture.F == 2:
    plotHGMdata(mixture, prefix=experiment, std=2)
    plotHGMclasses(mixture, prefix=experiment, std=1, n=1)

timer.add(f'{mixture._info()}\n\nCreated data')

accuracies = plotAccuracy(labels=mixture.labels(), 
                          prefix=experiment, 
                          k=10)

WSDM = GaussianWassersteinDistance(mixture.data_estimates())
timer.add('Computed distance matrices')

for w in range(n_plots):
    w      = round(w/(n_plots-1),2)
    info   = (mixture.C, mixture.N, w, seed, experiment)
    
    embedding = plotTSNE(TSNE, WSDM.matrix(w=w), info=info, sklearn=sklearn)
    timer.add(f'Done TSNE with sklearn={sklearn}')
    
    acc = accuracies.append(embedding, w)
    timer.add(f'Accuracy (w={w}): {acc}%')
    # plotMDS(MDS, matrix, info=info)
    # timer.add('Done MDS')

accuracies.plot()

timer.finish()