#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:19:26 2021

@author: fsvbach
"""

from Experiments.Visualization import Analysis 
from Datasets.BIG5 import BIG, Labels, Questions

dataset = BIG()
labels  = Labels()

sizes = dataset.groupby(level=0).mean()

Analysis._config.update(folder='continents', 
                       seed=13, 
                       name='countries',
                       description='',
                       dataset='Big5',
                       size= (1,10),
                       w=0, 
                       renaming= lambda name: Questions()[name])

# fig = Analysis.WassersteinEmbedding(dataset, labels, name='continents')
# fig.savefig("Reports/Figures/BIG5/EmbeddingContinents.pdf")

# identity = {cn: cn for cn in dataset.index.unique()}
# fig = Analysis.WassersteinEmbedding(dataset, identity, 
#                                     folder='flags', size= (2,10))
# fig.savefig("Reports/Figures/BIG5/EmbeddingCountries.pdf")

# fig = Analysis.SpecialCovariances(dataset, labels)
# fig.savefig("Reports/Figures/BIG/Covariances.pdf")

# fig = Analysis.Correlations(dataset, labels, normalize=False)
# fig.savefig("Reports/Figures/GER/Correlation.pdf")

features = ['OPN9', 'EXT6', 'EST7']
means = dataset[features].groupby(level=0).mean()
figure = Analysis.Features(dataset, labels, means, selection=True, suffix='Means')
figure.savefig("Reports/Figures/BIG5/FeatureMeans1.pdf")

features = ['AGR2', 'CSN7', 'OPN8']
means = dataset[features].groupby(level=0).mean()
figure = Analysis.Features(dataset, labels, means, selection=True, suffix='Means')
figure.savefig("Reports/Figures/BIG5/FeatureMeans2.pdf")
