#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

suffix = ''
nonvoters = True
if nonvoters:
    suffix += 'with Nonvoters'
title  = f'Political Landscape of Baden-Württemberg {suffix}'

from WassersteinTSNE import NormalTSNE
from Experiments.Visualization.Legend import plotElection
from Datasets.BW2021 import Wahlkreise

import matplotlib.pyplot as plt

BW = Wahlkreise(nonvoters=nonvoters)
dataset = BW.data.iloc[:,:7]

tsne = NormalTSNE()
embedding = tsne.fit(BW.data)


figure = plotElection(dataset, embedding, BW.average, 'Parteifarben')

fig, ax = plt.subplots(figsize=(10,10))

figure.show(ax)

ax.set_title(title, fontdict={'fontsize': 25})

fig.tight_layout()
fig.savefig(f'Plots/infoBW2021_NW{nonvoters}.svg')

fig.savefig(f'Reports/Figures/assets/infoBW2021_NW{nonvoters}.pdf')
plt.show()
plt.close()


    
    
    # dataset = Gemeinden(nonvoters=True)
    
    # tsne = WassersteinTSNE(seed=13, load='Data/Election BW2021/GemeindeEmbeddingTrue.npy')
    # embedding = tsne.fit(dataset.data.to_numpy())
    
    # figure = plotElection(dataset)
    # figure.tSNE(embedding, size=60)

# run()