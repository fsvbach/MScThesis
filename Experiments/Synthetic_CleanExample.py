#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:41:27 2021

@author: fsvbach
"""

from WassersteinTSNE import HierarchicalGaussianMixture as HGM
from WassersteinTSNE import AccuracyPlot, plotMixture, CleanExample


# mixture = HGM(seed=1)
# figure = plotMixture(mixture, std=2)
# figure.savefig("Plots/SyntheticHGM_RandomHGM.svg")
# figure.savefig("Reports/Figures/HGM/RandomHGM.pdf")

# mixture = CleanExample()
# figure = AccuracyPlot(mixture, k=15, seed=13)
# figure.savefig("Plots/SyntheticHGM_CleanExample.svg")
# figure.savefig("Reports/Figures/HGM/CleanExample.pdf")

mixture = HGM(seed=1, 
              datapoints=100, 
              classes=7,
              features=2,
              samples=15,
              ClassScaleVariance=1,
              ClassMeanDistance=2)
figure = AccuracyPlot(mixture, k=15, size=0.7, seed=13)
figure.savefig("Plots/SyntheticHGM_RandomExample.svg")
figure.savefig("Reports/Figures/HGM/RandomExample.pdf")

# toy = HGM(seed=13, 
#               datapoints=50, 
#               classes=3,
#               features=2,
#               samples=5,
#               ClassScaleVariance=1,
#               ClassMeanDistance=2)
# figure = AccuracyPlot(toy, lambdas=[0.1,0.9], size=2, seed=13, sklearn=True)
