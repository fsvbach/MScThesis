#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:24:39 2021

@author: fsvbach
"""

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import plotEVS
from Datasets.EVS2020.Data import Preprocess

import numpy as np
import matplotlib.pyplot as plt


EVS = Preprocess()
dataset, labels = EVS.NUTS(min_entries=2)

w = 1

fig, axes = plt.subplots(ncols=3, figsize=(45,10))

EVSGaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(EVSGaussians)
WT = WassersteinTSNE(WSDM)
embedding = WT.fit(w=w)
embedding.index = labels
embedding['sizes'] = 3
plotEVS(embedding, f'NORMAL embedding with w=1', ax=axes[0])
print('Plotted subplot')

EVSGaussians = Dataset2Gaussians(dataset, diagonal=True)
WSDM = GaussianWassersteinDistance(EVSGaussians)
WT = WassersteinTSNE(WSDM)
embedding = WT.fit(w=w)
embedding.index = labels
embedding['sizes'] = 3
plotEVS(embedding, f'embedding with DIAGONAL Covariance', ax=axes[1])
print('Plotted subplot')

EVSGaussians = Dataset2Gaussians(dataset, normalize=True)
WSDM = GaussianWassersteinDistance(EVSGaussians)
WT = WassersteinTSNE(WSDM)
embedding = WT.fit(w=w)
embedding.index = labels
embedding['sizes'] = 3
plotEVS(embedding, f'embedding with NORMALIZED Covariance', ax=axes[2])
print('Plotted subplot')

fig.suptitle('NUTS1 regions with Special Covariance', fontsize=30)  
fig.savefig(f'Plots/CovarianceAnalysis.svg')
plt.show()
plt.close() 