#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import numpy as np
from Code import Simulation, Distances, Plots
from Code.Visualization import plotHGM, plotTSNE
# from openTSNE import TSNE
# from openTSNE.sklearn import TSNE
from sklearn.manifold import TSNE

    
experiment = "TEST"

# N ~ Number of Datapoints (distributions)   
N = 30

# D ~ Average samples of each distribution
D = 50

# F ~ Number of Features of each sample
F = 2

# C ~ Number of classes
C = 3


mixture = Simulation.HierarchicalGaussian(datapoints=N, 
                                            samples=D, 
                                            features=F, 
                                            classes=C,
                                            ClassDistance=5,
                                            ClassVariance=30)

matrix = Distances.EuclideanDistanceMatrix(mixture.data_means)

plotHGM(mixture)

plotTSNE(TSNE, mixture.data_means, prefix=experiment)

plotTSNE(TSNE, matrix, precomputed=True, prefix=experiment)
