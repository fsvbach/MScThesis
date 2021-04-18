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

    xmeans, ymeans = mixture.data.reshape((-1,2)).T
    plt.scatter(xmeans, ymeans, s=5, c=samplecolor)
    
    xmeans, ymeans = mixture.data_means().T
    plt.scatter(xmeans, ymeans, s=100, c=datacolor)

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

    ax.text(0, 1, mixture._info(), transform=ax.transAxes, fontsize=15, verticalalignment='top')
    plt.title(f"Hierarchical Gaussian Mixture ({prefix})")
    plt.savefig(f"Plots/{prefix}_{mixture.seed}_HGM.eps")
    plt.show()
    
def plotTSNE(TSNE, mixture, metric, w=0.5, prefix='TEST'):
    name='Wasserstein'
    if w == 1:
        name='Euclidean'
        
    matrix = metric(mixture, w=w)
    
    tsne = TSNE(metric='precomputed', initialization='spectral', negative_gradient_method='bh')
    embedding = tsne.fit(matrix)
    
    N = mixture.N
    for i in range(mixture.C):
        points = embedding[N*i:N*(i+1)]
        xmeans, ymeans = points.T
        plt.scatter(xmeans, ymeans, s=50)
    
    plt.title(f"{name} embedding (w={w})")
    plt.savefig(f"Plots/{prefix}_{name}_{w}.eps")
    plt.show()
    plt.close()
