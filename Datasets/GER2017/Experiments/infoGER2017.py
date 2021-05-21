#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:12:53 2021

@author: fsvbach
"""

title = 'Political Landscape of Germany 2017'
import matplotlib.pyplot as plt
import numpy as np

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Elections import plotElection
from Datasets.GER2017.Data import Wahlbezirke

GER = Wahlbezirke()

labeldict = GER.labels.Gebiet.to_dict()

Gaussians, Names = Dataset2Gaussians(GER.data)

WSDM = GaussianWassersteinDistance(Gaussians)

WT = WassersteinTSNE(WSDM, Names, seed=13)

embedding = WT.fit(w=0)
index    = embedding.index.to_series()
embedding.index = index.map(labeldict)

figure = plotElection(GER.data.groupby(level=0).mean(), embedding, GER.mean)

size = 15

fig, ax = plt.subplots(figsize=(size,size))


figure.show(ax, 
            size=size, 
            numparty=6, 
            legend=(10,800,0),
            xstretch=50,
            ystretch=50,
            barwidth=30)

ax.set_title(title, fontdict={'fontsize': 25})

fig.savefig(f'Plots/infoGER2021.svg')
plt.show()
plt.close()
