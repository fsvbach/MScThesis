#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:03:06 2021

@author: bachmafy
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance



A = np.array([5,5,0,0,0])
B = np.array([5,0,0,0,5])

P1 = A/A.sum()
P2 = B/B.sum()
P3 = np.ones(len(A))/len(A)

space = np.arange(len(A))

dist12 = round(wasserstein_distance(space, space, P1, P2),3)
dist32 = round(wasserstein_distance(space, space, P3, P2),3)
dist13 = round(wasserstein_distance(space, space, P1, P3),3)

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))

ax1.bar(space, P1, alpha=0.3, label='P1')
ax2.bar(space, P2, alpha=0.3, label='P2')
ax3.bar(space, P3, alpha=0.3, label='P3')

# ax.set(xticks=space,
#        title ='Wasserstein Distance')
# ax.legend()

fig.savefig('Plots/WassersteinExperiments1D.svg')
# fig.savefig('Reports/Figures/Wasserstein/Experiments1D.pdf')
plt.show()
plt.close()