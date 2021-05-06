#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

from Code.TSNE import myTSNE
from Code.Loader import Gemeinden, Wahlkreise
from Code.Visualization import PlotElection


# dataset = Wahlkreise(nonvoters=True)

# tsne = myTSNE(load='Data/BW/KreiseEmbeddingTrue.npy')
# embedding = tsne.fit(dataset.data.to_numpy())

# figure = PlotElection(dataset)
# figure.tSNE(embedding)



dataset = Gemeinden(nonvoters=True)

tsne = myTSNE(seed=13, load='Data/BW/GemeindeEmbeddingTrue.npy')
embedding = tsne.fit(dataset.data.to_numpy())

figure = PlotElection(dataset)
figure.tSNE(embedding, size=60)

