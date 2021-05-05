#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:57:17 2021

@author: fsvbach
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

question1 = 'v199'
question2 = 'v187'
label = 'c_abrv'
group = 'v275b_N2'

df = pd.read_stata("Data/EVS.dta", convert_categoricals=False)
keys = pd.read_csv('Data/keys.csv', sep=';', header=None, index_col=0)[1].to_dict()

df.loc[df[question1]<0, question1] = np.NaN
df.loc[df[question2]<0, question2] = np.NaN

fig, ax = plt.subplots(figsize=(15,15))

# ax.scatter(df[question1], df[question2])

classes = df[[group, label, question1, question2]].groupby(label)
legend = []

for label, C in classes:
    
    rgb = np.random.rand(3,)
    # ax.scatter(C[question1].mean(), C[question2].mean(), c=[rgb], label=label, s=100)
    legend.append(mpatches.Patch(color=rgb, label=label))
    
    datapoints = C[[group, question1, question2]].groupby(group)

    for point, D in datapoints:
        
        ax.scatter(D[question1].mean(), D[question2].mean(), c=[rgb], s=20)
        
        
ax.legend(handles=legend, bbox_to_anchor=[1.0, 0.5], loc='center left')
fig.savefig('Plots/test.svg')
plt.show()
plt.close()
        