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
from Experiments.Visualization.ElectionPlot import plotElection
from Datasets.GER2017 import Bundestagswahl


GER = Bundestagswahl(numparty=6)
labels  = GER.labeldict('Gebiet')
dataset = GER.data

Gaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = WassersteinTSNE(WSDM, seed=13)
embedding = WT.fit(w=0.75)

index    = embedding.index.to_series()
embedding.index = index.map(labels)

mean = GER.subtract_mean()

figure = plotElection(GER.data.groupby(level=0).mean(), embedding, mean)

size = 60

fig, ax = plt.subplots(figsize=(size,size))

figure.show(ax, 
            size=size,
            legend=(2,5,700,20),
            xstretch=50,
            ystretch=80,
            barwidth=20,
            barheight=100,
            label=True)

ax.set_title(title, fontdict={'fontsize': 100})
        
fig.savefig(f'Plots/infoGER_transform.svg')
plt.show()
plt.close()
