#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:12:59 2021

@author: fsvbach
"""

from WassersteinTSNE.Distributions import WishartDistribution, RandomGenerator, CovarianceMatrix
from scipy.stats import wishart, ortho_group, special_ortho_group

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

_info = {'stretch': 0.8,
         'samples': 20,
         'columns': 6,
         'distance': 10,
         'name': '',
         'suffix':'',
         'seed':None,
         'nrows': 4}
         
def add_row(ax, CovarianceMatrices, info):
    dist = info['distance']
    
    generator = info['generator']
    n = info['columns']
    
    kind = 'Wishart'
    suffix = ' from ' + info['name'] +' Covariance'
    if info['name'] == 'Uniform':
        suffix = ''
        kind = 'Uniform'
        
    for d, A, i in zip(info['xvalues'], CovarianceMatrices, range(n)):
        color = 'C'+str(i)
        mean   = np.array([i*dist,0])
        
        if kind == 'Uniform':
            BBB  = [generator.UniformCovariance(2, d) for i in range(info['samples'])]
        else:
            Wishart = WishartDistribution(d, A)
            BBB  = generator.WishartSamples(Wishart, size=info['samples'])
            

        for B in BBB:
            width, height, angle = B.shape(std=1)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                           edgecolor='grey', facecolor='none', linewidth=1, 
                           label='c', linestyle='-')
            ax.add_patch(ell)
        
        if kind == 'Wishart':
            width, height, angle = A.shape(std=1)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                               edgecolor=color, facecolor='none', linewidth=4, 
                               label='c', linestyle='-')
            ax.add_patch(ell)
            
    ax.set_aspect('equal')
    ax.set(xbound=(-dist/2,(n-0.5)*dist),
           xticks=np.linspace(0,(n-1)*dist,num=n),
           xticklabels=info['xlabels'],
           yticks=np.linspace(-2*dist/3,2*dist/3,num=5))
    ax.set_title(f'{kind} Samples{suffix}', {'fontsize': 30})
    ax.set_xlabel('n', {'fontsize': 20})
    
            
    
def run(**kwargs):
    _info.update(kwargs)
    
    _info['generator'] = RandomGenerator(_info['seed'])
    generator = _info['generator']
    
    n = _info['columns']
    a = _info['stretch']
    nrows = _info['nrows']
    
    fig, ax = plt.subplots(nrows=nrows, figsize=(23,30))
    
    dist = 7*a
    _info.update(distance= dist,
                 xlabels  = np.linspace(0,(n-1)*dist,num=n)/dist+2,
                 xvalues  = np.arange(2,2+n),
                 name     ='Other')
    add_row(ax[2], [generator.UniformCovariance(2, 1)]*n, _info)

    
    dist = 10*a
    _info.update(distance= dist,
                name     ='Eye')
    add_row(ax[1], [CovarianceMatrix(np.eye(2), np.ones(2))]*n, _info)
    
    # add_row(ax   = ax[4], 
    #         ddd  = [2]*n,
    #         xlabel = [2]*n,
    #         AAA  =  [generator.UniformCovariance(2, 1) for i in range(n)],
    #         dist = 6*a,
    #         name = 'Random',
    #         samples=samples)
    
    dist = 10*a
    _info.update(distance= dist,
                 xlabels  = np.arange(1,1+n/2, 0.5),
                 xvalues  = np.arange(1,1+n/2, 0.5),
                 name     ='Uniform')
    add_row(ax[0], ['']*n, _info)


    dist = 9*a
    s    = np.array([3,0.5])
    OrthogonalMatrices  = generator.OrthogonalMatrix(dim=2, size=n)
    _info.update(distance= dist,
             xlabels  = [2]*n,
             xvalues  = [2]*n,
             name     ='Rotation')
    add_row(ax[3], [CovarianceMatrix(P, s) for P in OrthogonalMatrices], _info)
    
    fig.tight_layout()
    fig.savefig(f"Plots/CovarianceSampling{_info['suffix']}.svg")
    
if __name__ == '__main__':
    run(suffix='TEST')
