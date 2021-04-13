#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code import Simulations, Distances, Plots
from openTSNE import TSNE
import numpy as np
    
ND = Simulations.GaussianDistribution
CV = Simulations.CovarianceMatrix
WS = Distances.GaussianWasserstein

# N ~ Number of Datapoints (distributions)   
N = 30

# D ~ Average samples of each distribution
D = 50

# F ~ Number of Features of each sample
F = 2

# C ~ Number of classes
C = 3


G = [ND(np.zeros(F), CV(F, maxstd=10)) for i in range(2)]

A,B = G

d = WS(A,B)


mixture = Simulations.HierarchicalGaussian(datapoints=N, 
                                            samples=D, 
                                            features=F, 
                                            classes=C,
                                            ClassDistance=5,
                                            ClassVariance=30)

Plots.GaussianMixturePlot(mixture)



# tsne = TSNE()

# embedding = tsne.fit(mixture.data_means)

# xmeans, ymeans = embedding.T
# plt.scatter(xmeans, ymeans, s=50)
# plt.show()


