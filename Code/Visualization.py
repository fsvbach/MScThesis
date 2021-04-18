#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:12:56 2021

@author: fsvbach
"""

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
   
def plotHGM(mixture, prefix='TEST'):
    
    classcolor = 'red'
    datacolor = 'blue'
    samplecolor = 'green'
    
    fig = plt.figure(figsize=(16,16))
    ax = plt.subplot(111, aspect='equal')

    for i in range(mixture.C):
        GaussianClass = mixture.classes[i]
        
        xmean, ymean = GaussianClass.mean.T
        plt.scatter(xmean, ymean, s=400, c=classcolor)
        
        mean, width, height, angle = GaussianClass.shape()
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor=classcolor, facecolor='none', linewidth=1, linestyle='--')
        ax.add_artist(ell)
        
        GaussianData = mixture.datapoints[i*mixture.N]
        
        mean, width, height, angle = GaussianData.shape()
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                   edgecolor=datacolor, facecolor='none', linewidth=2 )
        ax.add_artist(ell)
        
    xmeans, ymeans = mixture.data_means().T
    plt.scatter(xmeans, ymeans, s=100, c=datacolor)

    xmeans, ymeans = mixture.data.reshape((-1,2)).T
    plt.scatter(xmeans, ymeans, s=5, c=samplecolor)
    
    plt.title(f"{prefix} Hierarchical Gaussian Mixture")
    plt.savefig(f"Plots/{prefix}_{mixture.seed}_HGM.eps")
    plt.show()
    
def plotTSNE(TSNE, matrix, name, prefix='TEST'):

    tsne = TSNE(metric='precomputed', initialization='spectral', negative_gradient_method='bh')
    embedding = tsne.fit(matrix)
    
    xmeans, ymeans = embedding.T
    plt.title(f"{name} embedding")
    plt.scatter(xmeans, ymeans, s=50)
    plt.savefig(f"Plots/{prefix}_{name}.eps")
    plt.show()
    plt.close()
