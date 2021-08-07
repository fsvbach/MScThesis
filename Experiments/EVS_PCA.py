#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 10:27:35 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Datasets.EVS2020 import EuropeanValueStudy
from Experiments.Visualization import utils

EVS = EuropeanValueStudy()
labeldict = EVS.labeldict()
    
A = EVS.data.values
B = A - A.mean(axis=0).reshape(1,-1)
C = B.T@B
s, P = np.linalg.eig(C)
plt.plot(s)
plt.show()
plt.close()

D = B @ P[:, :2] + A.mean(axis=0).reshape(1,-1)[:,:2]
U = pd.DataFrame(D, index=EVS.data.index, columns=['x','y'])
plt.scatter(U.x, U.y)
plt.show()
plt.close()

embedding = U.groupby(level=0).mean()
embedding.index = embedding.index.to_series(name='flags').map(labeldict)
embedding['sizes'] = 5
    
fig, ax = plt.subplots(figsize=(20,20))
utils.embedFlags(embedding, title="", ax=ax)
fig.savefig("Plots/EVS_PCA.svg")
plt.show()
plt.close()

fig, ax = plt.subplots()
for i in range(5):
    k = P[i]
    l = EVS.data.columns[np.argmax(np.abs(k))]
    ax.plot(k, label=l)
ax.legend(bbox_to_anchor=(1,1))
plt.show()
plt.close()


embedding = EVS.data.loc[:,['v201','v106']].groupby(level=0).mean()
embedding.index = embedding.index.to_series(name='flags').map(labeldict)
embedding.columns=['x','y']
embedding['sizes'] = 5
    
fig, ax = plt.subplots(figsize=(20,20))
utils.embedFlags(embedding, title="", ax=ax)
fig.savefig("Plots/EVS_v201v106.svg")
plt.show()
plt.close()
