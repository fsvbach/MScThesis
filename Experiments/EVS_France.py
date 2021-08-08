#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:09:34 2021

@author: fsvbach
"""

import numpy as np
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

A = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)
idxA = A.index.str.contains('FR')
FA = A.loc[idxA,idxA]
im = ax1.imshow(FA)
plt.colorbar(im, ax=ax1)
ax1.set_title('Exact Wasserstein')

Gaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(Gaussians)
B = pd.DataFrame(WSDM.matrix(w=0.5), index=WSDM.index, columns=WSDM.index)
idxB = WSDM.index.str.contains('FR')
FB = B.loc[idxB,idxB]
im = ax2.imshow(FB)
plt.colorbar(im, ax=ax2)
ax2.set_title('Gaussian Wasserstein')

fig.savefig('Plots/EVS_France.svg')
plt.show()
plt.close()

allA = A.loc['FRL0'].sort_values()
allB = B.loc['FRL0'].sort_values()

