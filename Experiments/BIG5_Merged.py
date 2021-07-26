#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:34:04 2021

@author: fsvbach
"""

from Datasets.BIG5 import Merged, Labels, Questions
from Experiments.Visualization import utils, Analysis

import matplotlib.pyplot as plt

dataset = Merged()
labels  = Labels()

# fig, ax = plt.subplots()

# utils.MeanStdCorr(dataset, ax=ax, title=r'Merged Treats')

# fig.savefig('Plots/BIG_VarStab.svg')
# plt.show()
# plt.close()


sizes = dataset.groupby(level=0).mean()
Analysis._config.update(folder='continents', 
                       seed=13, 
                       name='countries',
                       description='merged',
                       dataset='Big5',
                       size= (1,10),
                       w=0)

# fig = Analysis.WassersteinEmbedding(dataset, labels, name='continents')

fig = Analysis.Correlations(dataset, labels, normalize=False)

# means = dataset.groupby(level=0).mean()
# figure = Analysis.Features(dataset, labels, means, selection=False, suffix='Means')

# std = dataset.groupby(level=0).std()
# figure = Analysis.Features(dataset, labels, std, selection=False, suffix='Stds')
