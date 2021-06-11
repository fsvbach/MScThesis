#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:50:53 2021

@author: fsvbach
"""

from openTSNE import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from WassersteinTSNE.Visualization import embedScatter, plotMixture
from WassersteinTSNE import Dataset2Gaussians, GaussianWassersteinDistance, WassersteinTSNE, _naming
from WassersteinTSNE import HierarchicalGaussianMixture as HGM

toy = HGM(seed=13, 
              datapoints=50, 
              classes=3,
              features=2,
              samples=5,
              ClassScaleVariance=1,
              ClassMeanDistance=2)

fig, axes = plt.subplots(1, 5, figsize=(15,3))

Gaussians = Dataset2Gaussians(toy.data)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = WassersteinTSNE(WSDM, seed=13)
tsne = TSNE(metric='precomputed', 
          initialization='random', 
          negative_gradient_method='bh',
          random_state=13)
                   
### Embed five ScatterPlots
for ax, w in zip(axes, np.linspace(0,1,5)):

    embedding = tsne.fit(WSDM.matrix(w=w))
    embedding =  pd.DataFrame(embedding, 
                     index=WSDM.index,
                     columns = ['x','y'])
    # embedding = WT.fit(w=w)
    
    embedding.index = embedding.index.to_series().map(toy.labeldict())
    embedScatter(embedding, rf"{_naming.get(w, '')} embedding ($\lambda$={w})",
                 size=5, ax=ax)

fig.tight_layout()
fig.savefig('Plots/Synthetic_Alignment.svg')
plt.show()


