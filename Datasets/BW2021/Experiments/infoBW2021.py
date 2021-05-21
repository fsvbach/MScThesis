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
from WassersteinTSNE.Visualization.Elections import plotElection
from Datasets.BW2021.Data import Wahlkreise

import matplotlib.pyplot as plt

BW = Wahlkreise(nonvoters=nonvoters)

tsne = NormalTSNE()
embedding = tsne.fit(BW.data)


figure = plotElection(BW.data, embedding, BW.average)

fig, ax = plt.subplots(figsize=(15,15))

# figure.show(ax, numparty=7, legend=(3,200,0), )
figure.show(ax, numparty=7)

ax.set_title(title, fontdict={'fontsize': 25})

fig.savefig(f'Plots/infoBW2021_NW{nonvoters}.svg')
plt.show()
plt.close()


    
    
    # dataset = Gemeinden(nonvoters=True)
    
    # tsne = WassersteinTSNE(seed=13, load='Data/Election BW2021/GemeindeEmbeddingTrue.npy')
    # embedding = tsne.fit(dataset.data.to_numpy())
    
    # figure = plotElection(dataset)
    # figure.tSNE(embedding, size=60)

# run()