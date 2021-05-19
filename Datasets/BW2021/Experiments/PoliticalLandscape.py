#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

from WassersteinTSNE import NormalTSNE
from WassersteinTSNE.Visualization.Elections import plotElection
from Datasets.BW2021.Data import Wahlkreise

def run():
    dataset = Wahlkreise(nonvoters=True)
    
    tsne = NormalTSNE()
    embedding = tsne.fit(dataset.data.to_numpy())
    
    figure = plotElection(dataset)
    figure.tSNE(embedding)
    
    
    # dataset = Gemeinden(nonvoters=True)
    
    # tsne = WassersteinTSNE(seed=13, load='Data/Election BW2021/GemeindeEmbeddingTrue.npy')
    # embedding = tsne.fit(dataset.data.to_numpy())
    
    # figure = plotElection(dataset)
    # figure.tSNE(embedding, size=60)

run()