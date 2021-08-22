#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:34:14 2021

@author: fsvbach
"""

from WassersteinTSNE import NormalTSNE
from Experiments.Visualization.ElectionPlot import plotElection
from Datasets.BIG5 import Merged, Labels, Aligned
from Experiments.Visualization import utils, Analysis

import matplotlib.pyplot as plt

dataset = Merged()
labels  = Labels()
names = utils.code2name()

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(40,20))

means = dataset.groupby(level=0).mean()
tsne = NormalTSNE(seed=13)
embedding = tsne.fit(means)
embedding['sizes'] = 6
# embedding.index = embedding.index.to_series(name='continents').map(labels)
embedding.index.name='continents'
utils.embedFlags(embedding, 'Merged', ax2)

embedding = tsne.fit(Aligned().groupby(level=0).mean())
embedding['sizes'] = 6
embedding.index = embedding.index.to_series(name='continents').map(labels)
# embedding.index.name='continents'
utils.embedFlags(embedding, 'Aligned', ax1)

fig.savefig(f'Plots/BIG_ALIGNEDvsMERGED.svg')
plt.show()
plt.close()


average = means.mean(axis=0)
means -= average
figure = plotElection(means, embedding, average, 'Personality')

fig, ax = plt.subplots(figsize=(15,15))

figure.show(ax, xstretch=8)

ax.set_title('Psychological Landscape', fontdict={'fontsize': 25})

fig.savefig(f'Plots/infoBIG5.svg')

fig.savefig(f'Reports/Figures/BIG5/infoBIG5.pdf')
plt.show()
plt.close()


