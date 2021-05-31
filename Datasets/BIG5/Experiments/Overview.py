#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:43:48 2021

@author: fsvbach
"""

from Datasets.BIG5.Data import Aligned

import matplotlib.pyplot as plt

dataset = Aligned()

fig, axes = plt.subplots(10,12,figsize=(120,100))

for ax, (c, data) in zip(axes.flatten(), dataset.groupby(level=0)):
    C = data.corr()
    im = ax.imshow(C, vmin=-1, vmax=1, cmap='bwr')
    ax.set_title(c, fontsize=100)
    ax.axis('off')
    print('Plotted Heatmap')
    
    
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.set_ticks([-1,0,1])
cbar.set_ticklabels(['anti','none', 'high'])
cbar.ax.tick_params(labelsize=40)
fig.savefig(f'Plots/BIG5_Overview.png', dpi=50)
plt.show()
plt.close()    