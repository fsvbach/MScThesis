#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:25:29 2021

@author: bachmafy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Datasets.BIG5 import BIG, Labels
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import WassersteinDistanceMatrix
from WassersteinTSNE.TSNE import WassersteinTSNE
from Experiments.Visualization import utils

########### CALCULATING ################
name  = 'max500'
timer = Timer('BIG Exact Wasserstein')

dataset = BIG(maxnum=500)
labels = Labels()

K = WassersteinDistanceMatrix(dataset, timer=timer)
K.to_csv(f'Datasets/BIG5/Distances/{name}.csv')
np.save(f'Datasets/BIG5/Distances/{name}', K)  

timer.finish("Plots/.logfile.txt")

############## PLOTTING #################
A = pd.read_csv(f'Datasets/BIG5/Distances/{name}.csv', index_col=0)

tsne = WassersteinTSNE(seed=13)

embedding = tsne.fit(A)
embedding['sizes'] = 5
embedding.index =embedding.index.to_series(name='flags')

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(40,20))
utils.embedFlags(embedding, 'Exact Country embedding', ax=ax1)

embedding.index =embedding.index.to_series(name='continents').map(labels)
utils.embedFlags(embedding, 'Exact Continent embedding', ax=ax2)
fig.savefig(f'Plots/BIG_{name}_ExactWasserstein.svg')
plt.show()