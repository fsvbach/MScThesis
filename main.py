#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code import simulate
from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt

# N ~ Number of Datapoints (distributions)   
N = 200

# D ~ Average samples of each distribution
D = 50

# F ~ Number of Features of each sample
F = 2

# C ~ Number of classes
C = 5

mixture = simulate.GaussianMixture(datapoints=N, samples=D, features=F, classes=C)

xmeans, ymeans = mixture.class_means.T
plt.scatter(xmeans, ymeans, s=100)

xmeans, ymeans = mixture.data_means.T
plt.scatter(xmeans, ymeans, s=10)



