#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import WassersteinTSNE as ws

from WassersteinTSNE.Visualization.Synthetic import plotHGM
from WassersteinTSNE.TSNE import accuracy


def run(seed=None, suffix="TEST", k=10, sklearn=False, output=False, **kwargs): 

    
    timer      = ws.Timer(suffix, output=output)
          
    mixture = ws.HGM(seed=seed, **kwargs)
    
    if mixture.F == 2:
        fig, ax = plt.subplots(figsize=(15,10))
        ax = plotHGM(ax, mixture, std=3)
        fig.suptitle(mixture.info, fontsize=24)
        fig.savefig(f"Plots/Evalution{suffix}.svg")
        plt.show()
        plt.close()
        
    timer.add(f'{mixture._info()}\n\nCreated data')
    
    WT = ws.TSNE(mixture.data, seed=seed, fast_approx=False)
    timer.add('Computed distance matrices')
    
    fig, axes = plt.subplots(2,3, figsize=(15,10))
    
    w = 0
    accuracies = []
    for ax in axes.T.flatten()[:-1]:
        
        embedding = WT.fit(w=w)
        embedding.index = mixture.labels
    
        timer.add(f'Done TSNE with sklearn={sklearn}')
        
        ax.set(title=f"embedding (w={w})",
                aspect='equal')
        ax.axis('off')
        
        for c, coordinates in embedding.groupby(level=0):
            x,y = coordinates.T.values
            ax.scatter(x, y, s=1, c=f'C{c}')
        
        acc = accuracy(embedding)
        accuracies.append(acc*100)
          
        timer.result(f'Accuracy (w={w}): {acc}%')
        w += 0.25
    
    axes[1,2].plot(accuracies)
    axes[1,2].set(xlabel='w',
                    ylabel='%',
                    title =f"kNN Accuracies (k={k})",
                    ybound=(0,100))
                    # xbound=(0,1))
    
    
    # figure.plot()
    timer.result('Done Final Plot')
    
    timer.finish(f'Plots/.logfile.txt')
    
    plt.savefig("Plots/Evaluation.svg")
    plt.show()
    plt.close()

if __name__ == "__main__":
    run(suffix='TEST')

