#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""


from openTSNE import TSNE
from Code.EVS import EVS
from Code.Visualization import plotEVS

import pandas as pd
import numpy as np

dataset = EVS()

countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'ES', 'BG',  'FR']

labels = []
NUTS2  = []

for i, country in dataset.NUTS2(countries).groupby(level=0):
    for j, nuts in country.groupby(level=1):
        NUTS2.append(nuts.mean())
        # NUTS?cov.append(nuts.cov())
        labels.append(i)

data = pd.concat(NUTS2, axis=1).T

tsne = TSNE(random_state=13)

embedding = tsne.fit(data.to_numpy())

cord = pd.DataFrame(embedding, index=labels, columns=['x','y'])

plotEVS(cord)