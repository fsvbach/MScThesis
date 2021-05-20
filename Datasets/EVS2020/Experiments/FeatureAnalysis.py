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
from WassersteinTSNE.Visualization.Countries import plotEVS
from Datasets.EVS2020.Data import Preprocess


EVS = Preprocess()
dataset, labels = EVS.NUTS(min_entries=2)

# for i, (n, nuts) in enumerate(dataset.groupby(level=0)):
#     print(n,labels[i])
    
w = 0.5
# feature = 'v38'
EVSGaussians = Dataset2Gaussians(dataset)
WSDM = GaussianWassersteinDistance(EVSGaussians)
WT = WassersteinTSNE(WSDM, seed=13)

embedding = WT.fit(w=w)
embedding.index = labels
    
for feature in EVS.questions:
    fig, ax = plt.subplots(figsize=(15,10))
    
    sizes = dataset.groupby(level=0)[feature].mean().values
    minsize, maxsize = 1,5
    minval, maxval   = sizes.min(), sizes.max()
    sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
    embedding['sizes'] = sizes
    
    plotEVS(embedding, f'{EVS._info[feature][0]}', ax=ax)
    
    # fig.suptitle('NUTS1 regions from European countries', fontsize=30)  
    fig.savefig(f'Plots/Feature_{feature}.svg')
    plt.show()
    plt.close()        
    print('PLotted Feature')