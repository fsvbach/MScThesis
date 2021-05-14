#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

from Code.share.Simulations import CovarianceMatrix, GaussianDistribution
from Code.share.Wasserstein import WassersteinTSNE, GaussianWassersteinDistance
from Code.EVS2020 import DataLoader, Style

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



dataset = DataLoader.EVS()

countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'BG',  'FR']

labels = []
GaussianNUTS2  = []

for i, country in dataset.NUTS2().groupby(level=0):
    for j, nuts in country.groupby(level=1):
        size=len(nuts)
        if size > 1:
            mean = np.mean(nuts, axis=0)
            cov  = (nuts-mean).T@(nuts-mean)/(size-1)
            cov  = CovarianceMatrix(cov, from_array=True)
            GaussianNUTS2.append(GaussianDistribution(mean, cov))
            labels.append(i)

tsne = WassersteinTSNE(seed=13)

WSDM = GaussianWassersteinDistance(GaussianNUTS2)


w_range = [0,0.25,0.5,0.75,1]

num = len(w_range)
pic = 6


fig, axes = plt.subplots(ncols=num, figsize=(num*pic,pic))
    
for w, ax in zip(w_range, axes):
    
    embedding = tsne.fit_precomputed(WSDM.matrix(w=w))
    
    embedding = pd.DataFrame(embedding, index=labels, columns=['x','y'])

    Style.plotEVS(embedding, f'embedding with w={w}', ax=ax)
    print('Plotted subplot')
    
fig.suptitle('NUTS2 regions from European countries', fontsize=5*pic)  
fig.savefig(f'Plots/EVS_WasersteinTSNE.png')
plt.show()
plt.close()        
