#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

from .share.Distances import WassersteinTSNE
from .BW2021.DataLoader import Gemeinden, Wahlkreise
from .BW2021.Visualization import plotElection


def run():
    dataset = Wahlkreise(nonvoters=True)
    
    tsne = WassersteinTSNE(load='Data/Election BW2021/KreiseEmbeddingTrue.npy')
    embedding = tsne.fit(dataset.data.to_numpy())
    
    figure = plotElection(dataset)
    figure.tSNE(embedding)
    
    
    # dataset = Gemeinden(nonvoters=True)
    
    # tsne = WassersteinTSNE(seed=13, load='Data/Election BW2021/GemeindeEmbeddingTrue.npy')
    # embedding = tsne.fit(dataset.data.to_numpy())
    
    # figure = plotElection(dataset)
    # figure.tSNE(embedding, size=60)

