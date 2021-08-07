#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:09:34 2021

@author: fsvbach
"""

import pandas as pd
import matplotlib.pyplot as plt

from Experiments.Visualization.utils import embedFlags
from WassersteinTSNE import Dataset2Gaussians, GaussianTSNE, WassersteinTSNE, GaussianWassersteinDistance
from Datasets.EVS2020 import EuropeanValueStudy

name = 'complete'
kind = 'exact'

EVS = EuropeanValueStudy()
dataset = EVS.data
labels  = EVS.labels

A = pd.read_csv(f'Datasets/EVS2020/Distances/EVS_{name}.csv', index_col=0)

FR = A.index.str.contains('DE')

F = A.loc[FR,FR]

plt.imshow(F)