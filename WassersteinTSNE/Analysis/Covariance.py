#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:56:14 2021

@author: fsvbach
"""


from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import plotGER
from Datasets.GER2017.Data import Wahlbezirke

import numpy as np
import matplotlib.pyplot as plt


GER = Wahlbezirke(numparty=6)
namesdict =GER.labels.Bundesland.to_dict()
w = 1
size= 15

fig, axes = plt.subplots(ncols=3, figsize=(90,30))

Gaussians = Dataset2Gaussians(GER.data)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = WassersteinTSNE(WSDM, seed=13)
embedding = WT.fit(w=w)
index    = embedding.index.to_series()
embedding.index = index.map(namesdict)
embedding['sizes'] = size
plotGER(embedding, f'NORMAL embedding with w=1', ax=axes[0])
print('Plotted subplot')

Gaussians = Dataset2Gaussians(GER.data, diagonal=True)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = WassersteinTSNE(WSDM, seed=13)
embedding = WT.fit(w=w)
index    = embedding.index.to_series()
embedding.index = index.map(namesdict)
embedding['sizes'] = size
plotGER(embedding, f'embedding with DIAGONAL Covariance', ax=axes[1])
print('Plotted subplot')

Gaussians = Dataset2Gaussians(GER.data, normalize=True)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = WassersteinTSNE(WSDM, seed=13)
embedding = WT.fit(w=w)
index    = embedding.index.to_series()
embedding.index = index.map(namesdict)
embedding['sizes'] = size
plotGER(embedding, f'embedding with NORMALIZED Covariance', ax=axes[2])
print('Plotted subplot')

fig.suptitle('GER Wahlkreise with Special Covariance', fontsize=100)  
fig.savefig(f'Plots/GER_CovarianceAnalysis_max6.svg')
plt.show()
plt.close() 