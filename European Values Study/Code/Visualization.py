#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:47:11 2021

@author: bachmafy
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plotEVS(cord):
    plt.rcParams['legend.title_fontsize'] = 30
    fig, ax = plt.subplots(figsize=(20,20))
    for c, C in cord.groupby(level=0):
        ax.scatter(C['x'],C['y'], label=c)
    ax.legend(title='NUTS2 regions from', fontsize=30, markerscale=4, 
              ncol=2, bbox_to_anchor=(1.05,1.05))
    # ax.set_title('Citizens from', fontsize=40)
    fig.savefig('Plots/tsneEVS.svg')
    plt.show()
    plt.close()