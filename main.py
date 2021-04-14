#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

import numpy as np
from Code import Simulation, Distances
from Code.Visualization import plotHGM, plotTSNE
# from openTSNE import TSNE
# from openTSNE.sklearn import TSNE
from sklearn.manifold import TSNE

    
experiment = "TEST"

# C ~ Number of classes
C = 3

# N ~ Number of distributions per class (datapoints)   
N = 50

# D ~ Samples per Distribution
D = 50

# F ~ Number of Features of each sample
F = 2

mixture = Simulation.HierarchicalGaussian(datapoints=N, 
                                            samples=D, 
                                            features=F, 
                                            classes=C,
                                            ClassDistance=50,
                                            ClassVariance=50,
                                            DataVariance=5)

# matrix = Distances.EuclideanDistanceMatrix(mixture.data_means)
# matrix -= matrix.min()

matrix = Distances.WassersteinDistanceMatrix(mixture.data)

plotHGM(mixture, prefix=experiment)

plotTSNE(TSNE, mixture.data_means, prefix=experiment)

plotTSNE(TSNE, matrix, precomputed=True, prefix=experiment)
