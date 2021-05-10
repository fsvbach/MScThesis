#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from .share.Timer import Timer

from .share.Distances import GaussianWassersteinDistance
from .share.Simulations import HierarchicalGaussianMixture
from .share.Visualizations import plotHGM, plotTSNE


def run(): 
    n_plots    = 11
    seed       = 13
    experiment = "TEST"
    sklearn    = True
    timer      = Timer(experiment, output=True)
          
    
    mixture = HierarchicalGaussianMixture(seed=seed,
                                        datapoints=300, 
                                        samples=5, 
                                        features=2, 
                                        classes=7,
                                        ClassDistance=4,
                                        ClassVariance=6,
                                        DataVariance=5)
    
    if mixture.F == 2:
        plotHGM(mixture, prefix=experiment, std=2)
    timer.add(f'{mixture._info()}\n\nCreated data')
    
    WSDM = GaussianWassersteinDistance(mixture.data_estimates())
    timer.add('Computed distance matrices')
    
    
    figure = plotTSNE(labels=mixture.labels(), 
                      prefix=experiment, 
                      k=5)
    
    for w in range(n_plots):
        w      = round(w/(n_plots-1),2)
        info   = (mixture.C, mixture.N, w, seed, experiment)
        
        if sklearn:
            from sklearn.manifold import TSNE
            tsne = TSNE(metric='precomputed', 
                        square_distances=True, 
                        random_state=seed)
            embedding = tsne.fit_transform(WSDM.matrix(w=w))
        else:
            from openTSNE import TSNE
            tsne = TSNE(metric='precomputed', 
                        initialization='random', 
                        negative_gradient_method='bh',
                        random_state=seed)
            embedding = tsne.fit(WSDM.matrix(w=w))
        timer.add(f'Done TSNE with sklearn={sklearn}')
        
        acc = figure.append(embedding, w)
        timer.result(f'Accuracy (w={w}): {acc}%')
    
    figure.plot()
    timer.result('Done Final Plot')
    
    timer.finish(f'Plots/.logfile.txt')