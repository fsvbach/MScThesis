#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:04:56 2021

@author: bachmafy
"""

import matplotlib.pyplot as plt

from Experiments.Visualization.utils import embedFlags
from WassersteinTSNE import Dataset2Gaussians, GaussianTSNE, GaussianWassersteinDistance
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
dataset = EVS.data
labels  = EVS.labels

Gaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = GaussianTSNE(WSDM, seed=13)
embedding = WT.fit(w=0.5)

index    = embedding.index.to_series(name='flags')
embedding.index = index.map(labels)
embedding['name'] = index.values
embedding['sizes'] = 2

size = 20

fig, ax = plt.subplots(figsize=(size,size))

embedFlags(embedding, 'Names of NUTS regions', ax=ax)
ax.scatter(embedding['x'], embedding['y'])
for x, y, text in zip(embedding['x'], embedding['y'], embedding['name']):
    ax.text(x, y, text)
    print(text)

fig.savefig(f'Plots/EVS_info.svg')
plt.show()
plt.close()
