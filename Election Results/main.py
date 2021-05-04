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

nonvoters = True
newtSNE   = False
seed      = 13
numparty  = 7

farben = pd.read_csv('Data/BW/Farben.csv', delimiter=';', encoding='utf-8', header=0)
farben = farben.iloc[0].to_dict()

data = pd.read_csv('Data/BW/Kreise.csv', delimiter=';', encoding='ISO-8859-1', index_col='Wahlkreis')
head = 'Land Baden-Württemberg'
  
title = 'Political landscape of Baden-Württemberg'
total = data['Gültige Stimmen'] 
if nonvoters:
    total = data['Wahlberechtigte']
    data['Nichtwähler'] = total - data['Gültige Stimmen'] 
    farben['Nichtwähler'] = 'grey'
    title += ' with Non-Voters'
    
data.drop(columns=data.columns[:8], inplace=True)
data.fillna(0, inplace=True)
data.sort_values(head, axis=1, ascending=False, inplace=True)
 
data = data.divide(total, axis='rows')*100
average = data.loc[head]

data = (data-average)
data.drop(index=head, inplace=True)

def plotWahlkreis(district):
    name = ' '.join(data.loc[district].name.split(' ')[1:])
    fig, ax = plt.subplots()
    ax.bar(data.columns, data.loc[district])
    ax.set(ylabel='%',
           title=name)
    ax.tick_params(axis='x', labelrotation = 90)
    fig.savefig(f'Plots/Wahlkreise/{district}.png')
    plt.show()
    plt.close()

def plotBar(values, name):
    fig, ax = plt.subplots()
    ax.bar(data.columns, values)
    ax.set(ylabel='%',
           title=name)
    ax.tick_params(axis='x', labelrotation = 90)
    fig.savefig(f'Plots/Wahlkreise/{name}.png')
    plt.show()
    plt.close()

embedding = np.load(f'Data/BW/embedding{nonvoters}.npy')
if newtSNE:
    tsne = TSNE(random_state=seed)
    embedding = tsne.fit(data)
X,Y = embedding.T
    
fig, ax = plt.subplots(figsize=(15,15))

# box of wahlkreis result
f = 30
g = 50
k = 20
s = 3

l = 3

dist = k/numparty
for i in range(numparty):
    party = data.columns[i]
    color = farben[party]
    offset = (numparty/2-i)*dist-dist/2
    ax.bar(f*X-offset, s*data[party], bottom=g*Y, 
                           width=dist, color=color, align='center')
    
    ax.bar(f*min(X)+l*dist*i, average[i]*s, bottom=g*min(Y),
           color=color, width=l*dist, align='edge')

for name, x, y in zip(data.index, X, Y):
    text = ' '.join(name.split(' ')[1:])
    sub = data.loc[name].max()
    ax.annotate(text, (f*x-dist/2, g*y+sub*s), ha='center')
    

for spine in ax.spines:
    ax.spines[spine].set(visible=False)
ax.set(xticks=[],
       yticks=[])
ax.set_title(title, fontdict={'fontsize': 25})
fig.savefig(f'Plots/Wahlkreise-NW{nonvoters}.svg')
plt.show()