#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
from WassersteinTSNE import HGM 
from WassersteinTSNE.Visualization.Synthetic import plotHGM

def run():
    mixture = HGM(  seed=1,
                    datapoints=100, 
                    samples=30, 
                    features=2, 
                    classes=4,
                    ClassDistance=50,
                    DataVariance=5)
    
    fig, ax = plt.subplots(figsize=(15,10))
        
    ax = plotHGM(ax, mixture, std=3)

    fig.suptitle(mixture._info(), fontsize=24)

    fig.savefig(f"Plots/CleanExample.svg")
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    run()
