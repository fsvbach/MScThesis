#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:24:59 2021

@author: fsvbach
"""
# 
# from sklearn.manifold import TSNE
from openTSNE import TSNE
import numpy as np

matrix = np.genfromtxt('matrix.csv', delimiter=',')

tsne = TSNE(metric='precomputed', initialization='spectral', negative_gradient_method='fft')
embedding = tsne.fit(matrix)

# tsne = TSNE(metric='precomputed')
# embedding = tsne.fit_transform(matrix)


# NOT NECESSARY FOR MEMORY CRASH

import matplotlib.pyplot as plt


N = 50
for i in range(4):
    points = embedding[N*i:N*(i+1)]
    xmeans, ymeans = points.T
    plt.scatter(xmeans, ymeans, s=1)

plt.savefig("openTSNE_fft.png")
plt.show()
plt.close()