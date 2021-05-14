#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from WassersteinTSNE import Timer, WassersteinTSNE, HGM, GWD
from WassersteinTSNE.Visualization import plotHGM


def run(seed=None, n_plots=11, experiment="TEST", sklearn=False, output=True, **kwargs): 
    timer      = Timer(experiment, output=True)
          
    mixture = HGM(seed=seed, **kwargs)
    
    
    if mixture.F == 2:
        plotHGM(mixture, prefix=experiment, std=2)
    timer.add(f'{mixture._info()}\n\nCreated data')
    
    WSDM = GWD(mixture.data_estimates())
    timer.add('Computed distance matrices')
    
    
    figure = plotTSNE(labels=mixture.labels(), 
                      prefix=experiment, 
                      k=5)
    
    for w in range(n_plots):
        w      = round(w/(n_plots-1),2)
        info   = (mixture.C, mixture.N, w, seed, experiment)
        

        timer.add(f'Done TSNE with sklearn={sklearn}')
        
        acc = figure.append(embedding, w)
        timer.result(f'Accuracy (w={w}): {acc}%')
    
    figure.plot()
    timer.result('Done Final Plot')
    
    timer.finish(f'Plots/.logfile.txt')
    

