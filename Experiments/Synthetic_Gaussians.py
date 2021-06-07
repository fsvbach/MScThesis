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

def SqueezedGaussian(n=10):
    handles = []
    labels  = []

    fig, ax = plt.subplots(figsize=(5,5))
    for d in range(1,n):
        Gaussian = GaussianDistribution(mean=np.array([0,10*d]), cov=CovarianceMatrix(s=[n-d,d]))

        ell = plotGaussian(Gaussian, ax=ax, STDS=[4], size=5)
        ell.set_edgecolor('C'+str(d-1))
        ell.set_linestyle('solid')
            
        ax.axis('off')
        ax.set_aspect('equal')
        handles.append(ell)
        labels.append(f'd={d}')
    
    # adding legends
    ax.legend(handles, labels, 
              handler_map={Ellipse: HandlerEllipseRotation()},
              bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.savefig('Plots/SyntheticHGM_SqueezedGaussians.svg')

def RotatedGaussian(n=10):
    handles = []
    labels  = []

    fig, ax = plt.subplots(figsize=(5,5))
    for d in range(n):
        Gaussian = GaussianDistribution(mean=np.array([0,10*d]), cov=CovarianceMatrix(RotationMatrix(360/n*d)))
        
        ell = plotGaussian(Gaussian, ax=ax, STDS=[4], size=5)
        ell.set_edgecolor('C'+str(d))
        ell.set_linestyle('solid')
            
        ax.axis('off')
        ax.set_aspect('equal')
        handles.append(ell)
        labels.append(f'd={360/n*d}')
    
    # adding legends
    ax.legend(handles, labels, 
              handler_map={Ellipse: HandlerEllipseRotation()},
              bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.savefig('Plots/SyntheticHGM_RotatedGaussians.svg')


def SummaryGaussians(n=10):
    N,M = get_rectangle(n)
    fig, axes = plt.subplots(N,M, figsize=(5*M,5*N))
    
    for degree, ax in zip(np.linspace(0, 360, N*M), axes.flatten()):
        Gaussian = GaussianDistribution(mean=np.zeros(2), cov=CovarianceMatrix(RotationMatrix(degree)))
        
        ell = plotGaussian(Gaussian, ax=ax, STDS=range(1,5), size=40)
        ell.set_edgecolor('C1')
        ell.set_linestyle('solid')
            
        ax.axis('off')
        ax.set_aspect('equal')
        # adding legends
        ax.legend([ell], ['Covariance'], handler_map={Ellipse: HandlerEllipseRotation()})
        # ax.add_artist(leg1)
    
    fig.savefig('Plots/SyntheticHGM_SummaryGaussians.svg')
    
SqueezedGaussian()
RotatedGaussian()