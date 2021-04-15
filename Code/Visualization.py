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

    xmeans, ymeans = mixture.class_means.T
    plt.scatter(xmeans, ymeans, s=400, c=classcolor)

    for c in mixture.classes:
        mean, (width,height), angle = c.shape()
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                   edgecolor=classcolor, facecolor='none', linewidth=1, linestyle='--')
        ax.add_artist(ell)
        
    xmeans, ymeans = mixture.data_means.T
    plt.scatter(xmeans, ymeans, s=100, c=datacolor)
    
    for d in mixture.examples:
        mean, (width,height), angle = d.shape()
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                   edgecolor=datacolor, facecolor='none', linewidth=2 )
        ax.add_artist(ell)
        
    xmeans, ymeans = mixture.samples.T
    plt.scatter(xmeans, ymeans, s=5, c=samplecolor)
    
    plt.title(f"{prefix} Hierarchical Gaussian Mixture")
    plt.savefig(f"Plots/{prefix}_HierachicalGaussianMixture.eps")
    plt.show()
    
def plotTSNE(TSNE, data, precomputed=False, prefix='TEST'):
    metric = 'euclidean'
    # neighbors = 'auto'
    name   = "Euclidean"
    if precomputed:
        metric = 'precomputed'
        # neighbors = 'exact'
        name   = "Wasserstein"
        
    tsne = TSNE(metric=metric)#, neighbors=neighbors)#, square_distances=True)
    # embedding = tsne.fit(data)
    embedding = tsne.fit_transform(data)
    
    xmeans, ymeans = embedding.T
    plt.title(f"{name} embedding")
    plt.scatter(xmeans, ymeans, s=50)
    plt.savefig(f"Plots/{prefix}_{name}.eps")
    plt.show()
    plt.close()
