#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np

from WassersteinTSNE.Visualization.utils import embedScatter
from WassersteinTSNE.Visualization.Synthetic import plotHGM
from WassersteinTSNE import WassersteinTSNE as TSNE
from WassersteinTSNE import Timer, Dataset2Gaussians, GaussianWassersteinDistance 


def AccuracyPlot(synthetic, 
                 sklearn=False, 
                 output=False,
                 suffix='TEST',
                 seed=None,
                 n=5,
                 k=10):
    
    timer     = Timer(suffix, output=output)
          
    fig, axs = plt.subplots(2, 3, 
                             figsize=(40,20),
                             gridspec_kw={'width_ratios': [2, 1,1]})
    
    w0, w5, w10, acc = axs[0:,1:].flatten()
    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0:, 0]:
        ax.remove()
    hgm = fig.add_subplot(gs[0:, 0])
    
    
    plotHGM(hgm, synthetic.mixture, std=3)
    
    hgm.set_title(synthetic.mixture.info, fontsize=24)
        
    timer.add(f'{synthetic.mixture._info()}\n\nCreated data')
    
    WSDM = synthetic.WSDM()
    WT = TSNE(WSDM, seed=seed)
    timer.add('Computed distance matrices')
     
    ### Embed three ScatterPlots
    for ax, w in zip([w0,w5,w10], [0,0.5,1]):
        embedding = WT.fit(w=w)
        embedding.index = embedding.index.to_series().map(synthetic.mixture.labeldict())
        embedScatter(embedding, f"embedding (w={w})", ax)
        timer.add(f'Done TSNE with sklearn={sklearn}')
    
    ### Calculate Accuracies
    accuracies = []
    for w in np.linspace(0,1,n):
        accuracy = WT.accuracy(w, synthetic.mixture.labeldict(), k=k, n=n)
        accuracies.append(accuracy*100) 
        timer.result(f'Accuracy (w={w}): {accuracy}%')
    
    acc.plot(np.linspace(0,1,n), accuracies)
    acc.set(xlabel='w',
            ylabel='%',
            title =f"kNN Accuracies (k={k})",
            ybound=(0,100))
    
    timer.result('Done Final Plot')
    timer.finish(f'Plots/.logfile.txt')
    
    plt.savefig(f"Plots/AccuracyPlot_{suffix}.svg")
    plt.show()
    plt.close()

if __name__ == "__main__":
    from Datasets.SyntheticData import CleanExamples
    synthetic = CleanExamples.Load('Distinct')
    AccuracyPlot(synthetic, suffix='TEST')

