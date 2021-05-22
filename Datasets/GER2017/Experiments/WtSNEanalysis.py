#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:19:55 2021

@author: fsvbach
"""

import random
import matplotlib.pyplot as plt
import numpy as np

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import plotGER, plotImages
from Datasets.GER2017.Data import Wahlbezirke

GER = Wahlbezirke()

namesdict = GER.labels.Bundesland.to_dict()
 
Gaussians, names = Dataset2Gaussians(GER.data)
WSDM = GaussianWassersteinDistance(Gaussians)
WT = WassersteinTSNE(WSDM, names, seed=13)


fig, axes = plt.subplots(1,1, figsize=(50,50))
    
w = 0
ax=axes
# for ax in axes.T.flatten():
    # 
embedding = WT.fit(w=w)
index    = embedding.index.to_series()
embedding.index = index.map(namesdict)
embedding['sizes'] = 20

plotGER(embedding, title=f"embedding (w={w})", ax=ax)

w += 0.5

# h1, l1 = axes[0,0].get_legend_handles_labels()
# fig.legend(h1, l1, loc='lower right',
#            markerscale=5., scatterpoints=3, fontsize=15)
# axes[1,2].axis('off')

plt.savefig("Plots/GER_Wahlkreise.svg")
plt.show()
plt.close()
