#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:43 2021

@author: bachmafy
"""

from Experiments.Visualization import Analysis 
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy()
labels  = EVS.labeldict()
dataset = EVS.data


Analysis._config.update(folder='flags', 
                        seed=13, 
                        name='NUTS regions',
                        description='',
                        dataset='EVS',
                        renaming= lambda name: EVS.overview[name][0],
                        size= (3,15),
                        w=0.5)

figure = Analysis.WassersteinEmbedding(dataset, labels)

