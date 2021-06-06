#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:27:24 2021

@author: fsvbach
"""

from Experiments.Visualization import Analysis 
from Datasets import EVS2020 as Data

countries=None
trafo=False
NUTS=1

dataset, labels = Data.LoadEVS(Data.overview, countries=countries, transform=trafo, NUTS=NUTS, min_entries=2)

sizes = dataset.groupby(level=0).mean()

Analysis._config.update(folder='flags', 
                       seed=13, 
                       name=f'NUTS{NUTS} regions',
                       description='',
                       dataset='EVS',
                       renaming= lambda name: Data.overview[name][0],
                       size= (1,9))


# fig.suptitle(f'NUTS{NUTS} regions with Logit-Transformation: {trafo}', fontsize=30)  
# fig.savefig(f'Plots/NUTS{NUTS}RegionsTrafo{trafo}{suffix}.svg')
