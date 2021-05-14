#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:52:19 2021

@author: fsvbach
"""

import numpy as np

from WassersteinTSNE import Timer, GWD, RandomGenerator, GaussianDistribution
from WassersteinTSNE.Visualization.Heatmaps import plotMatrix

def run(seed=None, n=20, output=False, suffix=''):
    
    timer     = Timer(f'Frobenius{suffix}', output=output)
    generator = RandomGenerator(seed=seed)
    
    Gaussians = [GaussianDistribution(np.zeros(2), 
                                      generator.UniformCovariance(2, 1)) 
                 for i in range(n)]
    timer.add(f'create {n} Gaussians')
    
    FrobeniusDistance = GWD(Gaussians, fast_approx=True).matrix(w=1)
    timer.add('...calculated Frobenius Distance')
    
    ExactWasserstein  = GWD(Gaussians).matrix(w=1)
    timer.add('...calculated Wasserstein Distance')
    
    maxdiff = np.abs(ExactWasserstein - FrobeniusDistance)
    timer.result(f'Maxdiff: {np.max(maxdiff)}')
    
    plotMatrix([FrobeniusDistance, ExactWasserstein, maxdiff],
                ['FrobeniusDistance', 'ExactWasserstein', 'maxdiff'],
                name=f'Frobenius{suffix}')
    
    a, b = np.unravel_index(np.argmax(maxdiff, axis=None), maxdiff.shape)
    timer.result(f'Frob: {FrobeniusDistance[a,b]}')
    timer.result(f'Wass: {ExactWasserstein[a,b]}')
    
    A = Gaussians[a].cov
    B = Gaussians[b].cov
    
    frobdiff = np.linalg.norm(A.sqrt()-B.sqrt(), ord='fro')**2
    
    tmp = A.sqrt() @ B.array() @ A.sqrt()
    s, P = np.linalg.eig(tmp)
    tmp = A.array() + B.array() - 2 * P @ np.diag(np.sqrt(s)) @ P.T 
    wassdiff = np.sum(np.diag(tmp))
    
    timer.result(f'TestFrob: {frobdiff}')
    timer.result(f'TestWass: {wassdiff}')
    
    timer.finish(f'Plots/.logfile.txt')
    
if __name__ == '__main__':
    run(suffix='TEST')
