#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:58:18 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from WassersteinTSNE.Distances import GaussianWassersteinDistance, linprogSolver
from WassersteinTSNE.Distributions import GaussianDistribution, CovarianceMatrix, RotationMatrix
from Experiments.Visualization import utils

G1 = GaussianDistribution(mean= np.array([15,10]),
                          cov = CovarianceMatrix(RotationMatrix(45), s=[25,3]))
G2 = GaussianDistribution(mean= np.array([15,20]),
                          cov = CovarianceMatrix(RotationMatrix(-45), s=[50,5]))

# G1 = GaussianDistribution(mean= np.array([5,2]),
#                           cov = CovarianceMatrix(RotationMatrix(0), s=[4,0.001]))
# G2 = GaussianDistribution(mean= np.array([12,2]),
#                           cov = CovarianceMatrix(RotationMatrix(0), s=[49,0.001]))

WSDM = GaussianWassersteinDistance(pd.Series([G1,G2], index=['Blue', 'Orange']))

dist_e = WSDM.matrix(w=0)[1,0]
dist_c = WSDM.matrix(w=1)[1,0]
dist_w = WSDM.matrix()[1,0]

print(dist_e,dist_w, dist_c)

samplesizes = np.arange(1,9)*50
testsize = 20

results = np.zeros(shape=(len(samplesizes),testsize))
approx  = np.zeros(shape=(len(samplesizes),testsize))

for i, n in enumerate(samplesizes):
    print('Starting with', n)
    for j in range(testsize):
        U = G1.samples(size=n)
        V = G2.samples(size=n)
        opt_res = linprogSolver(U, V)
        results[i,j] = np.sqrt(-opt_res.fun )
        print(results[i,j])

# np.save('Experiments/Distances/GaussianDistances', results)
# results = np.load('Experiments/Distances/GaussianDistances.npy')
means = results.mean(axis=1)
stds  = results.std(axis=1)

fig, (ax, res) = plt.subplots(1, 2, figsize=(20,10))

utils.plotGaussian(G1, ax=ax, STDS=[2], color='C0', size=100)
utils.plotGaussian(G2, ax=ax, STDS=[2], color='C1', size=100)

ax.set_aspect('equal')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

ax.scatter(G1.mean[0], G1.mean[1], c='C1', s=200)
ax.scatter(G2.mean[0], G2.mean[1], c='C0', s=200)

res.plot(samplesizes, [dist_e for i in samplesizes],  color='C3', label='Distance of Means')
res.plot(samplesizes, 
         [dist_w for i in samplesizes],  
         color='C3', 
         linestyle='dashed',
         label='Gaussian Wasserstein Distance')
res.plot(samplesizes, [dist_c for i in samplesizes],  color='C3', label='Distance of Covariances')
res.plot(samplesizes, means, linewidth=5, color='C4', label='Estimated Wasserstein Distance')
res.fill_between(samplesizes, means+stds, means-stds, color='C4', alpha=0.3)

res.legend()
fig.savefig('Plots/WassersteinGaussian.svg')
plt.show()
plt.close()