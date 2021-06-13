#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:27:24 2021

@author: fsvbach
"""

from Experiments.Visualization.utils import MeanStdCorr
from Experiments.Visualization import Analysis 
from Datasets import EVS2020 as Data

import matplotlib.pyplot as plt

countries=None
NUTS=1

Analysis._config.update(folder='flags', 
                       seed=13, 
                       name=f'NUTS{NUTS} regions',
                       description='',
                       dataset='EVS',
                       renaming= lambda name: Data.overview[name][0],
                       size= (3,15))

dataset, labels = Data.LoadEVS(Data.overview, countries=countries, transform=True, NUTS=NUTS, min_entries=2) 

figure = Analysis.WassersteinEmbedding(dataset, labels, 
                              selection=[0,0.5,1], 
                              suffix=f'Logit')
figure.savefig("Reports/Figures/EVS/Transformation.pdf")






# fig, axes = plt.subplots(1,2, figsize=(10,5))

# for trafo, ax in zip([False,True], axes):
#     dataset, labels = Data.LoadEVS(Data.overview, countries=countries, transform=trafo, NUTS=NUTS, min_entries=2) 
#     MeanStdCorr(dataset, ax)
# 
# fig.savefig('Plots/EVS_VarStab.svg')
# plt.show()
# plt.close()   
