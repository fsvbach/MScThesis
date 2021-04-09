#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 11:15:50 2021

@author: fsvbach
"""

from Code import Simulate
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
C = 10

data = Simulate.GaussianMixture(datapoints=N, samples=D, features=F, classes=C)


