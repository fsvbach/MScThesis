#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:56:14 2021

@author: fsvbach
"""

from WassersteinTSNE import Analysis 
from Datasets.GER2017.Data import Wahlbezirke


GER       = Wahlbezirke(numparty=6).data
labeldict = GER.labels.Bundesland.to_dict()
dataset   = GER.data

sizes = GER.data.groupby(level=0).mean()

Analysis.config.update(folder='wappen', 
                       seed=13, 
                       name='Wahlkreise',
                       suffix='max6',
                       dataset='GER')

Analysis.WassersteinEmbedding(dataset, labeldict)

Analysis.SpecialCovariances(dataset, labels)

Analysis.Correlations(dataset, labels, 3, 5)

Analysis.Features(dataset, labels, sizes)