#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np

from .Visualization import embedScatter, plotMixture
from .TSNE import GaussianTSNE as TSNE
from .Distances import GaussianWassersteinDistance
from .Distributions import Dataset2Gaussians
from .utils import Timer, _naming

_config= {'sklearn': False, 
         'output':  True,
         'seed':    None,
         'lambdas': np.linspace(0,1,10),
         'k': 10,
         'std': 3,
         'size': 1}
    
def AccuracyPlot(mixture, **kwargs):
    config = {**_config, **kwargs}
    
    timer     = Timer('AccuracyPlot', output=config['output'])
          
    fig, axs = plt.subplots(2, 3, 
                             figsize=(20,10),
                             gridspec_kw={'width_ratios': [2, 1,1],
                                          'wspace': 0.05,
                                          'hspace':0.15})
    
    w0, w5, w10, acc = axs[0:,1:].flatten()
    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0:, 0]:
        ax.remove()
    hgm = fig.add_subplot(gs[0:, 0])
    
    plotMixture(mixture, std=config['std'], ax=hgm)
        
    timer.add(f'{mixture.info}\n\nCreated data')

    Gaussians = Dataset2Gaussians(mixture.data)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = TSNE(WSDM, seed=config['seed'], sklearn=config['sklearn'])
    timer.add('Computed distance matrices')
     
    ### Embed three ScatterPlots
    for ax, w in zip([w0,w5,w10], [0,0.5,1]):
        embedding = WT.fit(w=w)
        embedding.index = embedding.index.to_series().map(mixture.labeldict())
        embedScatter(embedding, rf"{_naming.get(w)} embedding ($\lambda$={w})",
                     size=config['size'], ax=ax)
        timer.add(f"Done TSNE with sklearn={config['sklearn']}")
    
    ### Calculate Accuracies
    accuracies = []
    for w in config['lambdas']:
        accuracy = WT.accuracy(w, mixture.labeldict(), k=config['k'])
        accuracies.append(accuracy*100) 
        timer.result(f'Accuracy (w={w}): {accuracy}%')
    
    acc.plot(config['lambdas'], accuracies, marker='o')
    acc.set(xlabel=r'$\lambda$',
            ylabel=r'$\%$',
            title =f"kNN Accuracies (k={config['k']})",
            ybound=(0,105),
            yticks=np.linspace(10,100,10))
    # # acc.tick_params(axis="y",direction="in", pad=-22)
    acc.yaxis.tick_right()
    acc.yaxis.set_label_position("right")
        
    timer.result('Done Final Plot')
    timer.finish('Plots/.logfile.txt')
    
    fig.tight_layout()
    return fig

