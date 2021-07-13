#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:25:29 2021

@author: bachmafy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
dataset = EVS.data
labels  = EVS.labels

A= pd.read_csv('Datasets/EVS2020/Distances/small.csv', index_col=0)

from WassersteinTSNE.TSNE import WassersteinTSNE

tsne =WassersteinTSNE(seed=13)
embedding = tsne.fit(A)

from Experiments.Visualization import utils

embedding['sizes'] = 5


embedding.index =embedding.index.to_series(name='flags').map(labels)

fig, ax = plt.subplots(figsize=(20,20))

utils.embedFlags(embedding, 'small', ax=ax)

fig.savefig('Plots/EVS_small.svg')
plt.show()