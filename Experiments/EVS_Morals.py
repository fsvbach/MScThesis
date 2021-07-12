#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:24:39 2021

@author: fsvbach
"""

from Experiments.Visualization import Analysis 
from Datasets.EVS2020 import EuropeanValueStudy

EVS = EuropeanValueStudy(morals=True)
dataset = EVS.data
labels  = EVS.labels

Analysis._config.update(folder='flags', 
                       seed=13, 
                       name='NUTS regions',
                       description='morals',
                       dataset='EVS',
                       renaming= lambda name: EVS.morals[name][0],
                       size= (3,15),
                       w=0.5)

# figure = Analysis.WassersteinEmbedding(dataset, labels)
# figure.savefig("Reports/Figures/EVS/Embedding.pdf")

# figure = Analysis.SpecialCovariances(dataset, labels)
# figure.savefig("Reports/Figures/EVS/Covariances.pdf")

# A = ['v187', 'v38', 'v103', 'v106', 'v201', 'v107', 'v200', 'v102']
# B = ['v143', 'v39', 'v102', 'v104', 'v188', 'v102', 'v63', 'v186']
# figure = Analysis.Correlations(dataset, labels, selection=zip(A,B), normalize=False)
# figure.savefig("Reports/Figures/EVS/Correlation.pdf")

means = dataset.groupby(level=0).mean()
figure = Analysis.Features(dataset, labels, means, selection=False, suffix='Means')
# figure.savefig("Reports/Figures/EVS/FeatureMeans.pdf")

stds = dataset.groupby(level=0).std()
figure = Analysis.Features(dataset, labels, stds, selection=False, suffix='Stds')
# figure.savefig("Reports/Figures/EVS/FeatureStds.pdf")
