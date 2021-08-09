#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:58:18 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import GaussianWassersteinDistance, linprogSolver
from WassersteinTSNE.Distributions import GaussianDistribution, CovarianceMatrix, RotationMatrix
from Experiments.Visualization import utils

timer = Timer("Gaussian Wasserstein")

G1 = GaussianDistribution(mean= np.array([15,10]),
                          cov = CovarianceMatrix(RotationMatrix(45), s=[25,3]))
G2 = GaussianDistribution(mean= np.array([15,20]),
                          cov = CovarianceMatrix(RotationMatrix(-45), s=[50,5]))

WSDM = GaussianWassersteinDistance(pd.Series([G1,G2], index=['Blue', 'Orange']))

dist_e = WSDM.matrix(w=0)[1,0]
dist_c = WSDM.matrix(w=1)[1,0]
dist_w = WSDM.matrix()[1,0]

samplesizes = np.arange(1,21)*50
testsize = 50

timer.add(f'Generated two Gaussians')

results = np.zeros(shape=(len(samplesizes),testsize))
cctimes = np.zeros(shape=(len(samplesizes),testsize))

for i, n in enumerate(samplesizes):
    for j in range(testsize):
        time = timer.time()
        U = G1.samples(size=n)
        V = G2.samples(size=n)
        opt_res = linprogSolver(U, V)
        results[i,j] = np.sqrt(-opt_res.fun )
        cctimes[i,j] = timer.time() - time
        print(results[i,j])
    timer.add(f'Computed {testsize} distances with {n} samples')

np.save('Experiments/Distances/GaussianDistances', results)
np.save('Experiments/Distances/GaussianTimes', timer)
# results = np.load('Experiments/Distances/GaussianDistances.npy')
# cctimes = np.load('Experiments/Distances/GaussianTimes.npy')

means = results.mean(axis=1)
stds  = results.std(axis=1)
times = cctimes.mean(axis=1)
tstds  = cctimes.std(axis=1)

timer.add(f'Saved distance matrix and times')

fig, (ax, res) = plt.subplots(1, 2, figsize=(20,10))

utils.plotGaussian(G1, ax=ax, STDS=[2], color='C0', size=100)
utils.plotGaussian(G2, ax=ax, STDS=[2], color='C1', size=100)

ax.set_aspect('equal')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

ax.scatter(G1.mean[0], G1.mean[1], c='C1', s=200)
ax.scatter(G2.mean[0], G2.mean[1], c='C0', s=200)

res.plot(samplesizes, [dist_e for i in samplesizes],  color='C3')
res.plot(samplesizes, 
         [dist_w for i in samplesizes],  
         color='C3', 
         linestyle='dashed',
         label='Gaussian Wasserstein Distance')
res.plot(samplesizes, [dist_c for i in samplesizes],  color='C3')
res.plot(samplesizes, means, linewidth=5, color='C4', label='Estimated Wasserstein Distance')
res.fill_between(samplesizes, means+stds, means-stds, color='C4', alpha=0.3)
res.set_ylabel('distance', color='C4')
res.set_xlabel('samplesize')

res2 = res.twinx()
res2.plot(samplesizes, times, linewidth=5, color='C6', label='Estimated Wasserstein Distance')
res2.fill_between(samplesizes, times+tstds, times-tstds, color='C6', alpha=0.3)
res2.set(ylabel='time')

res.legend()
fig.savefig('Plots/WassersteinGaussian.svg')
plt.show()
plt.close()

timer.add('Plotted Figure')
timer.finish('Plots/.logfile.txt')
