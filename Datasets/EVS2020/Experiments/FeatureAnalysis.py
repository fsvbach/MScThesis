#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:00:50 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import embedFlags
from Datasets.EVS2020 import Data

countries=None
trafo=False
NUTS=1
suffix=''

dataset, labels = Data.LoadEVS(Data.overview, countries=countries, transform=trafo, NUTS=NUTS, min_entries=2)
    
w = 0.5

EVSGaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(EVSGaussians)
WT = WassersteinTSNE(WSDM, seed=13)

embedding = WT.fit(w=w)
embedding.index = labels
    
for feature in dataset.columns:
    fig, ax = plt.subplots(figsize=(15,10))
    
    sizes = dataset.groupby(level=0)[feature].mean().values
    minsize, maxsize = 1,5
    minval, maxval   = sizes.min(), sizes.max()
    sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
    embedding['sizes'] = sizes
    
    embedFlags(embedding, f'{Data.overview[feature][0]}', 'flags', ax=ax)
  
    fig.savefig(f'Plots/Feature_{feature}.svg')
    plt.show()
    plt.close()        
    print('Plotted Feature')