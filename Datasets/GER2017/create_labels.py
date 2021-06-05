#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:34:46 2021

@author: fsvbach
"""

import pandas as pd

data = pd.read_csv('Datasets/GER2017/Data/GER Wahlkreise.csv', delimiter=';',
                encoding='utf-8', header=5, usecols=[0,1,2])
data = data.dropna()
data.Nr = data.Nr.astype('int')

mask = data['gehört zu'] == 99

## Länder
länder = data.loc[mask, ['Nr', 'Gebiet']]
länder.set_index('Nr', inplace=True)
länder = länder.Gebiet.to_dict()

### Kreise
kreise = data.loc[~mask]
kreise.set_index('Nr', inplace=True)
kreise['Label'] = kreise['gehört zu'].astype('int')

kreise['Bundesland'] = kreise['Label'].map(länder)
kreise.drop('gehört zu', axis=1, inplace=True)

### Store as DataFrame
kreise.to_csv("Datasets/GER2017/Data/labels.csv")