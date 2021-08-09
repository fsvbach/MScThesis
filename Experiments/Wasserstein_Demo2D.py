#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:56:30 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, wasserstein_distance
from WassersteinTSNE.Distances import GaussianWassersteinDistance, linprogSolver
from WassersteinTSNE.Distributions import GaussianDistribution, CovarianceMatrix, RotationMatrix
from Experiments.Visualization import utils

G1 = GaussianDistribution(mean= np.array([5,2]),
                          cov = CovarianceMatrix(RotationMatrix(0), s=[4,0.001]))
G2 = GaussianDistribution(mean= np.array([12,2]),
                          cov = CovarianceMatrix(RotationMatrix(0), s=[49,0.001]))

G1 = {'mu':5, 'sigma': 2}
G2 = {'mu':12, 'sigma': 7}

dist_e = np.abs(G1['mu'] - G2['mu'])
dist_c = np.abs(G1['sigma'] - G2['sigma'])
dist_w = np.sqrt(dist_e**2 + dist_c**2)

fig, axes = plt.subplots(2, 2, figsize=(20,20))

U = norm.rvs(loc=G1['mu'], scale=G1['sigma'], size=1000, random_state=13)
V = norm.rvs(loc=G2['mu'], scale=G2['sigma'], size=900, random_state=13)

a,b = min(min(U), min(V)), max(max(U),max(V))

x    = np.linspace(a,b, 100)
pdf1 = norm.pdf(x, loc=G1['mu'], scale=G1['sigma'])
pdf2 = norm.pdf(x, loc=G2['mu'], scale=G2['sigma'])

# n1, bins1, _ = ax.hist(U, density=True, alpha=0.5, bins=x)
# n2, bins2, _ = ax.hist(V, density=True, alpha=0.5, bins=x)
 
ax.bar(x, pdf1, alpha=0.5)
ax.bar(x, pdf2, alpha=0.5)
ax.plot(x, pdf1, color='C0')
ax.plot(x, pdf2, color='C1')

dist_exact = wasserstein_distance(x,x,pdf1,pdf2)

print(dist_e,dist_w, dist_c, dist_exact)
plt.show()