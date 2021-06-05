#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:56:14 2021

@author: fsvbach
"""

from WassersteinTSNE import Analysis 
from Datasets.GER2017.Data import Wahlbezirke


GER     = Wahlbezirke(numparty=6)
labels  = GER.labels.Bundesland.to_dict()
dataset = GER.data

sizes = GER.data.groupby(level=0).std()

Analysis._config.update(folder='wappen', 
                        seed=13, 
                        name='Wahlkreise',
                        description='max6',
                        dataset='GER')

# Analysis.WassersteinEmbedding(dataset, labels)

Analysis.WassersteinEmbedding(dataset, labels, 
                              selection=[0,0.5,0.75,0.875,0.9475,1], 
                              suffix='_long')

Analysis.SpecialCovariances(dataset, labels)

Analysis.Correlations(dataset, labels)
Analysis.Correlations(dataset, labels, normalize=False, suffix='_normalized')

# Analysis.Features(dataset, labels, sizes, suffix='_std')