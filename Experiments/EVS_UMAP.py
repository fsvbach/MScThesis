#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:21:04 2021

@author: bachmafy
"""

import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Datasets.EVS2020 import EuropeanValueStudy
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import WassersteinDistanceMatrix, GaussianWassersteinDistance
from WassersteinTSNE.TSNE import WassersteinTSNE, GaussianTSNE
from WassersteinTSNE.Distributions import RotationMatrix, MirrorMatrix, Dataset2Gaussians
from Experiments.Visualization import utils

name = 'complete2W'
EVS = EuropeanValueStudy(max_entries=1000)
labels  = EVS.labeldict()
dataset = EVS.data

A = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)

U = umap.UMAP(metric='precomputed', random_state=13)
embedding = U.fit_transform(A)

embedding =  pd.DataFrame( embedding, 
             index=A.index.to_series(name='flags').map(labels),
             columns = ['x','y'])
embedding['sizes'] = 5
        
fig, ax = plt.subplots(figsize=(20,20))
utils.embedFlags(embedding, "UMAP embedding", ax=ax)
fig.savefig("Plots/EVS_UMAP.svg")
