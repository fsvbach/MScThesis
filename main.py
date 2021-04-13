#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code import Simulations
# from openTSNE import TSNE
# from openTSNE.sklearn import TSNE
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

plot = "TEST"

# N ~ Number of Datapoints (distributions)   
N = 100

# D ~ Average samples of each distribution
D = 50

# F ~ Number of Features of each sample
F = 2

# C ~ Number of classes
C = 2

mixture = Simulations.HierarchicalGaussian(datapoints=N, 
                                           samples=D, 
                                           features=F, 
                                           classes=C,
                                           maxClassVariance=500,
                                           maxDataVariance=100)

fig = plt.figure(figsize=(16,16))

xmeans, ymeans = mixture.class_means.T
plt.scatter(xmeans, ymeans, s=200)

xmeans, ymeans = mixture.data_means.T
plt.scatter(xmeans, ymeans, s=50)

xmeans, ymeans = mixture.datapoints.T
plt.scatter(xmeans, ymeans, s=5)

plt.savefig(f"Plots/{plot}_HierachicalGaussianMixture.eps")
plt.show()
plt.close()

def tsnePlot(data, precomputed=False, prefix='TEST'):
    metric = 'euclidean'
    name   = "Euclidean"
    if precomputed:
        metric = 'precomputed'
        name   = "Wasserstein"
        
    tsne = TSNE(metric=metric)
    embedding = tsne.fit_transform(data)
    
    xmeans, ymeans = embedding.T
    plt.title(f"{name} embedding")
    plt.scatter(xmeans, ymeans, s=50)
    plt.savefig(f"Plots/{prefix}_{name}.eps")
    plt.show()
    plt.close()

tsnePlot(mixture.data_means, prefix=plot)

K = mixture.data_means @ mixture.data_means.T

tsnePlot(np.abs(K), precomputed=True, prefix=plot)
