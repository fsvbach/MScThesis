#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import plotEVS
from Datasets.EVS2020.Data import Preprocess

trafo = False
NUTS  = 2
countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'BG',  'FR', 'ES']

EVS = Preprocess(countries=None, transform=trafo)
dataset, labels = EVS.NUTS(NUTS=NUTS, min_entries=20)

# for i, (n, nuts) in enumerate(dataset.groupby(level=0)):
#     print(n,labels[i])
    
EVSGaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(EVSGaussians)
WT = WassersteinTSNE(WSDM, seed=13)

fig, axes = plt.subplots(ncols=3, figsize=(45,10))

for w, ax in zip([0,0.5,1], axes):
    
    embedding = WT.fit(w=w)
    embedding.index = labels
    embedding['sizes'] = 3
    plotEVS(embedding, f'embedding with w={w}', ax=ax)
    print('Plotted subplot')
    
fig.suptitle(f'NUTS{NUTS} regions with Logit-Transformation: {trafo}', fontsize=30)  
fig.savefig(f'Plots/NUTS{NUTS}RegionsTrafo{trafo}.svg')
plt.show()
plt.close()        
