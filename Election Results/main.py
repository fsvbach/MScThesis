#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openTSNE import TSNE

nonvoters = False

data = pd.read_csv('Data/BW/Kreise.csv', delimiter=';', encoding='ISO-8859-1', index_col='Wahlkreis')

total = data['Gültige Stimmen'] 
if nonvoters:
    total = data['Wahlberechtigte']
    data['Nichtwähler'] = total - data['Gültige Stimmen'] 
    
data = data.drop(columns=data.columns[:8]).divide(total, axis='rows').fillna(0)

average = data.loc['Land Baden-Württemberg']

data = data.drop(index='Land Baden-Württemberg') - average

def plotWahlkreis(district):
    name = ' '.join(data.loc[district].name.split(' ')[1:])
    fig, ax = plt.subplots()
    ax.bar(data.columns, data.loc[district])
    ax.set(ylabel='%',
           title=name)
    ax.tick_params(axis='x', labelrotation = 90)
    plt.show()
    plt.close()
    
# for i in data.index:
#     plotWahlkreis(i)

tsne = TSNE()

embedding = tsne.fit(data)
    
x,y =embedding.T

plt.scatter(x,y)