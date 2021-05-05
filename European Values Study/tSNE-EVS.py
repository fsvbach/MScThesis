#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

from sklearn.manifold import TSNE

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

questions = ['v38', 'v39',          # ability to change life
             'v63',                 # religion
             'v102', 'v103', 'v104', 'v105', 'v106', 'v107',  # social welfare
             'v143', 'v144',        # satisfaction with status quo
             'v185', 'v186', 'v187', 'v188',                # immigration
             'v199', 'v201','v202','v203','v200'           # environment
             ]

countries = ['DE', 'SE', 'IT', 'HU', 'GB']

label = 'c_abrv'
group = 'v275b_N2'

df = pd.read_stata("Data/EVS.dta", convert_categoricals=False, index_col='c_abrv')

data = df.loc[countries, questions]
data[data<0] = np.NaN
data = data.dropna()

tsne = TSNE(random_state=13)

embedding = tsne.fit_transform(data)

cord = pd.DataFrame(embedding, index=data.index, columns=['x','y'])

plt.rcParams['legend.title_fontsize'] = 30
fig, ax = plt.subplots(figsize=(20,20))
for c, C in cord.groupby(level=0):
    ax.scatter(C['x'],C['y'], label=c)
ax.legend(title='Citizens from', fontsize=30, markerscale=4)
# ax.set_title('Citizens from', fontsize=40)
fig.savefig('Plots/tsneEVS.svg')
plt.show()
plt.close()