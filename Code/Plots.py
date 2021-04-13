#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:12:56 2021

@author: fsvbach
"""

import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
   

def GaussianMixturePlot(mixture):
    
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
    
    plt.savefig("Plots/GaussianMixture.eps")
    plt.show()
