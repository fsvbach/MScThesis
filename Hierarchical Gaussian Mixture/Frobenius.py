#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:52:19 2021

@author: fsvbach
"""

import numpy as np
import matplotlib.pyplot as plt

from Code.utils import Timer
from Code.Distances import GaussianWassersteinDistance, WassersteinMatrixLoop
from Code.Simulation import Generator, GaussianDistribution
from Code.Visualization import plotHGM, plotMatrix

timer     = Timer('Frobenius', output=True)
generator = Generator(seed=13)

n = 2000

Gaussians = [GaussianDistribution(np.zeros(2), 
                                  generator.UniformCovariance(2, 1)) 
             for i in range(n)]
timer.add(f'create {n} Gaussians')

FrobeniusDistance = GaussianWassersteinDistance(Gaussians).matrix(w=1)
timer.add('...calculated Frobenius Distance')

ExactWasserstein  = WassersteinMatrixLoop(Gaussians, w=1)
timer.add('...calculated Wasserstein Distance')

maxdiff = np.abs(ExactWasserstein - FrobeniusDistance)
timer.result(f'Maxdiff: {np.max(maxdiff)}')

plotMatrix([FrobeniusDistance, ExactWasserstein, maxdiff],
            ['FrobeniusDistance', 'ExactWasserstein', 'maxdiff'],
            name='LargeFrobeniusExperiment')

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

timer.finish(f'Plots/Heatmaps/.logfile.txt')
