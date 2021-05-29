#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:23:31 2021

@author: fsvbach
"""

import random
import matplotlib.pyplot as plt
import numpy as np

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import plotGER, plotImages
from Datasets.GER2017.Data import Wahlbezirke


fig, axes = plt.subplots(1,3, figsize=(90,30))

for i, number in enumerate([10, 20, 30]):
    GER = Wahlbezirke(numparty=number)
    namesdict = GER.labels.Bundesland.to_dict()
 
    Gaussians = Dataset2Gaussians(GER.data, normalize=True)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=13)
    
    embedding = WT.fit(w=1)
    index    = embedding.index.to_series()
    embedding.index = index.map(namesdict)
    embedding['sizes'] = 10

    plotGER(embedding, title=f"N={number} parties", ax=axes[i])

# h1, l1 = axes[0,0].get_legend_handles_labels()
# fig.legend(h1, l1, loc='lower right',
#            markerscale=5., scatterpoints=3, fontsize=15)
# axes[1,2].axis('off')

plt.savefig("Plots/Number of Parties.svg")
plt.show()
plt.close()
