#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:19:26 2021

@author: fsvbach
"""

from WassersteinTSNE import Analysis 
from Datasets.BIG5.Data import Complete, Labels

dataset = Complete()
labels  = Labels()

sizes = dataset.groupby(level=0).mean()

Analysis.config.update(folder='continents', 
                       seed=13, 
                       name='country',
                       suffix='',
                       dataset='Big5',
                       size= (1,9))

Analysis.WassersteinEmbedding(dataset, labels)

# Analysis.SpecialCovariances(dataset, labels)

# Analysis.Correlations(dataset, labels, 3, 5)

# Analysis.Features(dataset, labels, sizes)