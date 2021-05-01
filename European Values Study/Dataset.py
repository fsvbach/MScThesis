#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 12:57:17 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

question1 = 'v102'
question2 = 'v103'
label = 'c_abrv'
group = 'v275b_N2'

df = pd.read_stata("Data/EVS.dta", convert_categoricals=False)

df.loc[df[question1]<0, question1] = np.NaN
df.loc[df[question2]<0, question2] = np.NaN

fig, ax = plt.subplots(figsize=(15,15))

ax.scatter(df[question1], df[question2])

classes = df[[group, label, question1, question2]].groupby(label)

for label, C in classes:
    
    rgb = np.random.rand(3,)
    ax.scatter(C[question1].mean(), C[question2].mean(), c=[rgb], label=label, s=100)
       
    datapoints = C[[group, question1, question2]].groupby(group)

    for point, D in datapoints:
        
        ax.scatter(D[question1].mean(), D[question2].mean(), c=[rgb], s=20)
        
ax.legend(bbox_to_anchor=[1.0, 0.5], loc='center left')
fig.savefig('Plots/political landscape.svg')
plt.show()
plt.close()
        