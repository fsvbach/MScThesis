#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:43:48 2021

@author: fsvbach
"""

from Datasets.BIG5 import Aligned
from Experiments.Visualization.utils import get_rectangle, code2name

import matplotlib.pyplot as plt

labels = code2name()

dataset = Aligned()

countries = dataset.index.unique()
countries = ['tz', 'de', 'us' , 've', 'ir', 'mm', 'cn', 'ru']
dataset = dataset.loc[dataset.index.isin(countries)]

m,n = get_rectangle(len(countries))

fig, axes = plt.subplots(m,n,figsize=(7*n,7*m))

for ax, (c, data) in zip(axes.flatten(), dataset.groupby(level=0)):
    C = data.corr()
    im = ax.imshow(C, vmin=-1, vmax=1, cmap='bwr')
    # ax.set_title(f'{c.upper()} ({len(data)})', fontsize=50)
    ax.set_title(f'{labels[c.upper()]} ({len(data)})', fontsize=20)
    ax.axis('off')
    print('Plotted Heatmap')
    
    
cbar = fig.colorbar(im, ax=axes.ravel().tolist())
cbar.set_ticks([-1,0,1])
cbar.set_ticklabels(['anti','none', 'high'])
cbar.ax.tick_params(labelsize=40)
fig.savefig('Plots/BIG5_Overview_small.png')

# fig.savefig('Reports/Figures/BIG5/Overview.pdf') 
