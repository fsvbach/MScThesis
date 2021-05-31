#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:24:39 2021

@author: fsvbach
"""

from WassersteinTSNE import Analysis 
from Datasets.EVS2020 import Data

countries=None
trafo=False
NUTS=1

dataset, labels = Data.LoadEVS(Data.overview, countries=countries, transform=trafo, NUTS=NUTS, min_entries=2)

sizes = dataset.groupby(level=0).mean()

Analysis.config.update(folder='flags', 
                       seed=13, 
                       name=f'NUTS{NUTS} regions',
                       suffix='',
                       dataset='EVS',
                       renaming= lambda name: Data.overview[name][0],
                       size= (1,9))

# fig.suptitle(f'NUTS{NUTS} regions with Logit-Transformation: {trafo}', fontsize=30)  
# fig.savefig(f'Plots/NUTS{NUTS}RegionsTrafo{trafo}{suffix}.svg')

Analysis.WassersteinEmbedding(dataset, labels)

Analysis.SpecialCovariances(dataset, labels)

A = ['v187', 'v38', 'v103', 'v106', 'v201', 'v107', 'v200', 'v102']
B = ['v143', 'v39', 'v102', 'v104', 'v188', 'v102', 'v63', 'v186']
Analysis.Correlations(dataset, labels, 2, 4, selection=zip(A,B), normalize=True)

Analysis.Features(dataset, labels, sizes, w=0)