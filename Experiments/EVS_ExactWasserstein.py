#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:25:29 2021

@author: bachmafy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import linprog

from Datasets.EVS2020 import EuropeanValueStudy
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import EuclideanDistance, SparseConstraint
from WassersteinTSNE.TSNE import WassersteinTSNE
from Experiments.Visualization import utils

timer = Timer('EVS Exact Wasserstein')

EVS = EuropeanValueStudy(max_entries=6000)
labels  = EVS.labeldict()
dataset = EVS.data

nuts = dataset.index.unique()
N = len(nuts)

K = np.zeros((N,N))

k = 0
for i in range(N):
    for j in range(i+1, N):
        
        U = dataset.loc[nuts[i]]
        V = dataset.loc[nuts[j]]
        
        D = EuclideanDistance(U, V)
        n, m = len(U), len(V)
    
        # timer.add(f'Computed {n}x{m} distance matrix of {nuts[i]} and {nuts[j]}')
        
        A = SparseConstraint(n,m)
        b = np.concatenate([np.ones(n)/n, np.ones(m)/m])
        c = D.reshape(-1)
        
        # timer.add(f'Created {n+m}x{n*m} constraint matrix')
        
        opt_res = linprog(-b, A.T, c, bounds=[None, None], method='highs')
        emd = -opt_res.fun
        
        K[i,j] = emd
        
        if k%250 == 0:
            timer.add(f'Completed {k} of {N*(N-1)/2}')
        k+=1

K = K + K.T

A = pd.DataFrame(K, 
                     index=nuts,
                     columns =nuts
                     )

A.to_csv('Datasets/EVS2020/Distances/big.csv')

np.save('Datasets/EVS2020/Distances/big', K)  

timer.finish("Plots/.logfile.txt")

############## PLOTTING #################


A= pd.read_csv('Datasets/EVS2020/Distances/big.csv', index_col=0)


tsne =WassersteinTSNE(seed=13)
embedding = tsne.fit(A)

embedding['sizes'] = 5


embedding.index =embedding.index.to_series(name='flags').map(labels)

fig, ax = plt.subplots(figsize=(20,20))

utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax)

fig.savefig('Plots/EVS_ExactWasserstein.svg')
plt.show()