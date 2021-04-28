#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:12:59 2021

@author: fsvbach
"""

from Code.Simulation import Generator, GaussianDistribution, CovarianceMatrix

from scipy.stats import wishart, ortho_group, special_ortho_group
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

generator = Generator(seed=12)
samples   = 20
n         = 6

def add_row(ax, ddd, xlabel, AAA, dist, name):
    kind = 'Wishart'
    suffix = ' from ' + name +' Covariance'
    if name == 'Uniform':
        suffix = ''
        kind = 'Uniform'
        
    for d, A, i in zip(ddd, AAA, range(n)):
        color = 'C'+str(i)
        m   = np.array([i*dist,0])
        
        if kind == 'Uniform':
            BBB  = [generator.UniformCovariance(2, d) for i in range(samples)]
        else:
            BBB  = wishart.rvs(d, A.array(), random_state=generator.NewSeed(), size=samples)
            

        for B in BBB:
            
            if kind == 'Wishart':
                s, P = np.linalg.eig(B)
                C = GaussianDistribution(m, CovarianceMatrix(P, s))
            else:
                C = GaussianDistribution(m, B)
                
            mean, width, height, angle = C.shape(std=1)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                           edgecolor='grey', facecolor='none', linewidth=1, 
                           label='c', linestyle='-')
            ax.add_patch(ell)
        
        if kind == 'Wishart':
            C = GaussianDistribution(m, A)
            mean, width, height, angle = C.shape(std=1)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                               edgecolor=color, facecolor='none', linewidth=4, 
                               label='c', linestyle='-')
            ax.add_patch(ell)
            
    ax.set_aspect('equal')
    ax.set(xbound=(-dist/2,(n-0.5)*dist),
           xticks=np.linspace(0,(n-1)*dist,num=n),
           xticklabels=xlabel,
           yticks=np.linspace(-2*dist/3,2*dist/3,num=5))
    ax.set_title(f'{kind} Samples{suffix}', {'fontsize': 30})
    ax.set_xlabel('n', {'fontsize': 20})

        
          
a=0.8
fig, ax = plt.subplots(nrows=5, figsize=(23,30))

dist = 7*a
add_row(ax   = ax[2], 
        ddd  = np.arange(2,2+n),
        xlabel = np.linspace(0,(n-1)*dist,num=n)/dist+2,
        AAA  = [generator.UniformCovariance(2, 1)]*n,
        dist = dist,
        name = 'Other')

dist = 10*a
add_row(ax   = ax[1], 
        ddd  = np.arange(2,2+n),
        xlabel = np.linspace(0,(n-1)*dist,num=n)/dist+2,
        AAA  = [CovarianceMatrix(np.eye(2), np.ones(2))]*n,
        dist = dist,
        name = 'Eye')

add_row(ax   = ax[4], 
        ddd  = [2]*n,
        xlabel = [2]*n,
        AAA  =  [generator.UniformCovariance(2, 1) for i in range(n)],
        dist = 6*a,
        name = 'Random')

add_row(ax   = ax[0], 
        ddd  = np.arange(1,1+n/2, 0.5),
        xlabel = np.arange(1,1+n/2, 0.5),
        AAA  =  ['']*n,
        dist = 10*a,
        name = 'Uniform')

s    = np.array([3,0.5])
PPP  = special_ortho_group.rvs(dim=2, random_state=generator.NewSeed(), size=n)
add_row(ax   = ax[3], 
        ddd  = [2]*n,
        xlabel = [2]*n,
        AAA  =  [CovarianceMatrix(P, s) for P in PPP],
        dist = 9*a,
        name = 'Rotation')

fig.tight_layout()
fig.savefig(f'Plots/CovarianceSampling.svg')
