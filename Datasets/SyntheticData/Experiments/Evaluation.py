#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import WassersteinTSNE as ws

from WassersteinTSNE.Visualization.Synthetic import plotHGM


def run(seed=None, suffix="TEST", sklearn=False, output=False, **kwargs): 
    timer      = ws.Timer(suffix, output=output)
          
    mixture = ws.HGM(seed=seed, **kwargs)
    
    if mixture.F == 2:
        fig, ax = plt.subplots(figsize=(15,10))
        ax = plotHGM(ax, mixture, std=3)
        fig.suptitle(mixture._info(), fontsize=24)
        fig.savefig(f"Plots/Evalution{suffix}.svg")
        plt.show()
        plt.close()
        
    timer.add(f'{mixture._info()}\n\nCreated data')
    
    WT = ws.TSNE(mixture.data, seed=seed, fast_approx=False)
    timer.add('Computed distance matrices')
    
    fig, axes = plt.subplots(2,3)
    
    w = 0
    for ax in axes.T.flatten()[:-1]:
        
        x,y = WT.fit(w=w)
        timer.add(f'Done TSNE with sklearn={sklearn}')
        
        ax.set(title=f"embedding (w={w})",
               aspect='equal')
        
        ax = plotEmbedding(ax, coordinates)
        
        timer.result(f'Accuracy (w={w}): {acc}%')
    
    ax[2,3].plot(accuracies)
    ax[2,3].legend()
    
    figure.plot()
    timer.result('Done Final Plot')
    
    timer.finish(f'Plots/.logfile.txt')

# if __name__ == "__main__":
#     run(suffix='TEST')

