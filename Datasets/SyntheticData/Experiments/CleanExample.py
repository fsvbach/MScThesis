#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

from WassersteinTSNE import HGM 
from WassersteinTSNE.Visualization.Gaussians import plotHGM

def run():
    mixture = HGM(  seed=1,
                    datapoints=50, 
                    samples=30, 
                    features=2, 
                    classes=4,
                    ClassDistance=50,
                    DataVariance=5)
    
    plotHGM(mixture, std=3, prefix='cleanExample')

if __name__ == '__main__':
    run()
