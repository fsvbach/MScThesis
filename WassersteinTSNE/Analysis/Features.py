#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:10:44 2021

@author: fsvbach
"""


from WassersteinTSNE import DataObject
from WassersteinTSNE.Visualization.Countries import plotGER
from Datasets.GER2017.Data import Wahlbezirke

import numpy as np
import matplotlib.pyplot as plt

GER = Wahlbezirke(numparty=6)
labeldict = GER.labels.Bundesland.to_dict()

WassersteinGER = DataObject(GER.data)

tsne = WassersteinGER.TSNE(seed=13)

embedding = tsne.fit(w=1)
embedding.index = embedding.index.to_series().map(labeldict)
  
for feature in WassersteinGER.dataset.columns:    
    sizes = WassersteinGER.dataset.groupby(level=0)[feature].mean().values
    # sizes = WassersteinGER.dataset.groupby(level=0)[feature].std().values
  
    minsize, maxsize = 5,25
    minval, maxval   = sizes.min(), sizes.max()
    sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
    embedding['sizes'] = sizes

    fig, ax = plt.subplots(figsize=(15,15))

    plotGER(embedding, f'feature {feature}', ax=ax)
    print('PLotted Feature')
 
    fig.savefig(f'Plots/Feature_{feature}.svg')
    plt.show()
    plt.close()    