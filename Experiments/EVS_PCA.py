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

fig, ax = plt.subplots()
A = EVS.data.values
B = A - A.mean(axis=0).reshape(1,-1)
C = B.T@B
s, P = np.linalg.eig(C)
ax.plot(s)
fig.savefig("Plots/EVS_PCA_Eigenvalues.svg")
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
utils.embedFlags(embedding, title="PCA embedding", ax=ax)
fig.savefig("Plots/EVS_PCA_embedding.svg")
plt.show()
plt.close()

fig, ax = plt.subplots()
for i in range(3):
    k = P[i]
    l = EVS.data.columns[np.argmax(np.abs(k))]
    ax.plot(k, label=rf'$v_{i+1}$ with max at {l}')
ax.legend(bbox_to_anchor=(1,1))
ax.set(title="Eigenvectors", xlabel="Feature")
fig.savefig(f"Plots/EVS_PCA_Eigenvectors.svg")
plt.show()
plt.close()

features = ['v107','v187']
embedding = EVS.data.loc[:,features].groupby(level=0).mean()
embedding.index = embedding.index.to_series(name='flags').map(labeldict)
embedding.columns=['x','y']
embedding['sizes'] = 5
    
fig, ax = plt.subplots(figsize=(20,20))
for label, data in embedding.groupby(level=0):
    X, Y, s = data['x'], data['y'], data['sizes']
    ax.scatter(X, Y,label=label, s=s/100)
    flag = plt.imread(f'{utils.PATH}/{embedding.index.name}/{label}.png')
    utils.plotImages(X, Y, flag, s, ax)

ax.set_title("Projection in two dimensions", fontsize=100)
ax.set_xlabel(EVS.legend[features[0]][0], fontsize=50)
ax.set_ylabel(EVS.legend[features[1]][0], fontsize=50)
fig.savefig(f"Plots/EVS_PCA_{''.join(features)}.svg")
plt.show()
plt.close()
