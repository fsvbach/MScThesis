#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:41:27 2021

@author: fsvbach
"""

from WassersteinTSNE import HierarchicalGaussianMixture as HGM
from WassersteinTSNE import AccuracyPlot, plotMixture, CleanExample

mixture = HGM(seed=1)
figure = plotMixture(mixture, std=2)
figure.savefig("Plots/SyntheticHGM_RandomHGM.svg")
figure.savefig("Reports/Figures/HGM/RandomHGM.pdf")

mixture = CleanExample()
figure = AccuracyPlot(mixture, n=10, k=15)
figure.savefig("Plots/SyntheticHGM_CleanExample.svg")
figure.savefig("Reports/Figures/HGM/CleanExample.pdf")
