#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code import Simulations
from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt

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
                                           maxClassVariance=1,
                                           maxDataVariance=1)

fig = plt.figure(figsize=(16,16))

xmeans, ymeans = mixture.class_means.T
plt.scatter(xmeans, ymeans, s=200)

xmeans, ymeans = mixture.data_means.T
plt.scatter(xmeans, ymeans, s=50)

xmeans, ymeans = mixture.datapoints.T
plt.scatter(xmeans, ymeans, s=5)

plt.savefig("Plots/GaussianMixture.eps")
plt.show()
plt.close()

tsne = TSNE()

embedding = tsne.fit(mixture.data_means)

xmeans, ymeans = embedding.T
plt.scatter(xmeans, ymeans, s=50)
plt.show()


