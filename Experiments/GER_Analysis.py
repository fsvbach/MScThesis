#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:56:14 2021

@author: fsvbach
"""

from Experiments.Visualization import Analysis 
from Datasets.GER2017 import Bundestagswahl

GER     = Bundestagswahl(numparty=6)
labels  = GER.labeldict()
dataset = GER.transform()

Analysis._config.update(folder='wappen', 
                        seed=13, 
                        name='Wahlkreise',
                        description='max6',
                        size=(10,30),
                        dataset='GER',
                        w=0.75)

fig = Analysis.WassersteinEmbedding(dataset, labels, 
                              selection=[0,0.5,0.75,0.875,0.9475,1], 
                              suffix='2rows')
# fig.savefig("Reports/Figures/GER/Embedding.pdf")

# fig = Analysis.SpecialCovariances(dataset, labels)
# fig.savefig("Reports/Figures/GER/Covariances.pdf")

# fig = Analysis.Correlations(dataset, labels, normalize=False)
# fig.savefig("Reports/Figures/GER/Correlation.pdf")

# features = ['CDU', 'DIE LINKE', 'GRÜNE']
# means = dataset[features].groupby(level=0).mean()
# figure = Analysis.Features(dataset, labels, means, selection=True, suffix='Means')
# figure.savefig("Reports/Figures/GER/FeatureMeans.pdf")

# stds = dataset[features].groupby(level=0).std()
# figure = Analysis.Features(dataset, labels, stds, selection=True, suffix='Stds')
# figure.savefig("Reports/Figures/GER/FeatureStds.pdf")
