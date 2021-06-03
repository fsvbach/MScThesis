#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

from Datasets.SyntheticData.CleanExamples import Create 
from WassersteinTSNE.Visualization.Synthetic import plotHGM
from WassersteinTSNE import Dataset2Gaussians, GaussianWassersteinDistance

import matplotlib.pyplot as plt
import pandas as pd

path = Create.path

class Load:
    functions = {'Distinct' : Create.Distinct,
                 'Random'   : Create.Random,
                 'DoubleCross'    : Create.DoubleCross}
    
    def __init__(self, name='Distinct'):
        self.name    = name
        self.mixture = self.functions[name]()
        self.dataset = pd.read_csv(f'{path}/{name}.csv', 
                                   index_col=(0,1))

    def WSDM(self):
        data = self.dataset.droplevel(0)
        Gaussians = Dataset2Gaussians(data)
        return GaussianWassersteinDistance(Gaussians)
    
    def Visualize(self):
        fig, ax = plt.subplots(figsize=(15,10))
            
        ax = plotHGM(ax, self.mixture, std=3)
    
        fig.suptitle(self.mixture._info(), fontsize=24)
    
        fig.savefig(f"Plots/CleanExample_{self.name}.svg")
        plt.show()
        plt.close()
