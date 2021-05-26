#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:10:44 2021

@author: fsvbach
"""


from WassersteinTSNE import Dataset2Gaussians, GaussianWassersteinDistance, WassersteinTSNE
from WassersteinTSNE.Visualization.Countries import plotGER
from Datasets.GER2017.Data import Wahlbezirke

import numpy as np
import matplotlib.pyplot as plt

w = 1
n = 20

GER = Wahlbezirke(numparty=n)
labeldict = GER.labels.Bundesland.to_dict()

Gaussians = Dataset2Gaussians(GER.data, normalize=True)
WSDM = GaussianWassersteinDistance(Gaussians)
tsne = WassersteinTSNE(WSDM, seed=13)

embedding = tsne.fit(w=w)
embedding.index = embedding.index.to_series().map(labeldict)

fig, axes = plt.subplots(10,19, figsize=(190,100))

axes = axes.flatten()
idx = 0     
for i, feature in enumerate(GER.data.columns):    
    # sizes = WassersteinGER.dataset.groupby(level=0)[feature].mean().values
    # sizes = WassersteinGER.dataset.groupby(level=0)[feature].std().values
    
    for j, feature2 in enumerate(GER.data.columns[:i]):
        corr = GER.data.groupby(level=0).corr().fillna(0)
        sizes = corr.swaplevel().loc[feature, feature2].values
        
        # minsize, maxsize = 5,25
        # minval, maxval   = sizes.min(), sizes.max()
        # sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
        # embedding['sizes'] = sizes
        

        # plotGER(embedding, f'correlation {feature}{feature2}', ax=ax)
        i-=1
        im=axes[idx].scatter(embedding['x'], embedding['y'],
                   c=sizes, cmap='seismic', vmax=1, vmin=-1)
        axes[idx].set_title(f'{feature} with {feature2}',
                            fontsize=25)
        axes[idx].axis('off')
        # fig.colorbar(m, ax=ax[i,j])

        idx+=1
        print('PLotted Feature')
     
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.set_ticks([-1,0,1])
cbar.set_ticklabels(['anti', 'none', 'high'])
cbar.ax.tick_params(labelsize=60)
fig.suptitle('Correlation among parties', fontsize=100)  
fig.savefig(f'Plots/Feature_correlations.svg')
plt.show()
plt.close()    