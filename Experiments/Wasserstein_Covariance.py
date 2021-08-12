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
from WassersteinTSNE.Distributions import arr2cov, GaussianDistribution, CovarianceMatrix, RotationMatrix
from Experiments.Visualization import utils

y = 5
G = GaussianDistribution(mean= np.array([0,0]),
                         cov = CovarianceMatrix(RotationMatrix(0), s=[10,1]))

Gref = GaussianDistribution(mean= np.array([0,y]),
                            cov = CovarianceMatrix(RotationMatrix(0), s=[10,1]))

covariances = [CovarianceMatrix(RotationMatrix(90), s=[10,1]),
               CovarianceMatrix(RotationMatrix(90), s=[0.5,0.05]),
               CovarianceMatrix(RotationMatrix(30), s=[20,1]),
               CovarianceMatrix(RotationMatrix(0), s=[1,1])]

def PairwiseCovarianceDistance(cov1, cov2):
    tmp = cov2.sqrt() @ cov1.array() @ cov2.sqrt()
    tmp = arr2cov(tmp)
    tmp = cov1.array() + cov2.array() - 2 * tmp.sqrt()
    return np.sum(np.diag(tmp))

distances = []
for cov in covariances:
    distances.append(PairwiseCovarianceDistance(cov, G.cov))

fig, ax = plt.subplots(figsize=(10,10))

mean, width, height, angle = G.shape(std=1)
ell = utils.Ellipse(xy=mean, width=width, height=height, angle=angle, 
              edgecolor='black', facecolor='none', 
              linewidth=10)
ax.add_patch(ell)

mean, width, height, angle = Gref.shape(std=1)
ell = utils.Ellipse(xy=mean, width=width, height=height, angle=angle, 
              edgecolor='black', facecolor='none', 
              linewidth=3)
ax.add_patch(ell)

Gaussians = []
for i, dist in enumerate(distances):
    color = 'C'+str(i)
    mean = RotationMatrix(90*i) @ np.sqrt([0,y**2-dist])
    Gaussians.append(GaussianDistribution(mean, covariances[i]))
    # circle = plt.Circle((0,0), dist, color='C'+str(i), fill=False)
    # ax.add_patch(circle)
    ax.scatter(mean[0],mean[1], c=color, s=20)
    utils.plotGaussian(Gaussians[-1], size=0, STDS=[1], color=color, ax=ax)
    
ax.set_aspect('equal')

fig.savefig("Plots/Wasserstein_Covariance.svg")