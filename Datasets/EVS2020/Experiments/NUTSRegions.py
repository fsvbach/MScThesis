#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from WassersteinTSNE import WassersteinTSNE
from Datasets.EVS2020.Data import Preprocess
from WassersteinTSNE.Visualization.Countries import plotEVS

countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'BG',  'FR', 'ES']

EVS = Preprocess()
dataset, labels = EVS.NUTS2( min_entries=2)

# for i, (n, nuts) in enumerate(dataset.groupby(level=0)):
#     print(n,labels[i])
    
WT = WassersteinTSNE(dataset)

w = 0.5
fig, ax = plt.subplots(figsize=(15,10))
    
embedding = WT.fit(w=w)
embedding.index = labels

plotEVS(embedding, f'embedding with w={w}', ax=ax, size = 3)
print('Plotted subplot')
    
fig.suptitle('NUTS2 regions from European countries', fontsize=30)  
fig.savefig(f'Plots/NUTSRegions.svg')
plt.show()
plt.close()        
