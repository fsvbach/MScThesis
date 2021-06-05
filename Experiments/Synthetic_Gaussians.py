#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:05:40 2021

@author: fsvbach
"""

from WassersteinTSNE.Distributions import GaussianDistribution, RotationMatrix, CovarianceMatrix
from WassersteinTSNE.Visualization import HandlerEllipseRotation, Ellipse, HandlerEllipse
from Experiments.Visualization.utils import plotGaussian, get_rectangle

import numpy as np
import matplotlib.pyplot as plt

handles = []
labels  = []

n = 10
N,M = get_rectangle(n)

fig, axes = plt.subplots(N,M, figsize=(5*M,5*N))

for degree, ax in zip(np.linspace(0, 360, N*M), axes.flatten()):
    
    Gaussian = GaussianDistribution(mean=np.zeros(2), cov=CovarianceMatrix(RotationMatrix(degree)))
    
    ell = plotGaussian(Gaussian, ax=ax, size=40)
    
    ell.set_edgecolor('C1')
    ell.set_linestyle('solid')
        
    ax.axis('off')
    ax.set_aspect('equal')
    # adding legends
    ax.legend([ell], ['Covariance'], handler_map={Ellipse: HandlerEllipseRotation()})
    # ax.add_artist(leg1)
    
fig.savefig('Plots/SyntheticHGM_Gaussians.svg')