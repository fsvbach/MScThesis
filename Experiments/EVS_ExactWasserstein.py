#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:25:29 2021

@author: bachmafy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Datasets.EVS2020 import EuropeanValueStudy
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import WassersteinDistanceMatrix
from WassersteinTSNE.TSNE import WassersteinTSNE
from Experiments.Visualization import utils

from WassersteinTSNE import Dataset2Gaussians, GaussianTSNE, GaussianWassersteinDistance, _naming, RotationMatrix

def calculate(name, maxnum=10000):
    
    ########### CALCULATING ################
    timer = Timer('EVS Exact Wasserstein')
    
    EVS = EuropeanValueStudy(max_entries=maxnum)
    labels  = EVS.labeldict()
    dataset = EVS.data
    
    K = WassersteinDistanceMatrix(dataset, timer=timer)
    K.to_csv(f'Experiments/Distances/EVS_{name}.csv')
    # np.save(f'Datasets/EVS2020/Distances/{name}', K)  
    
    timer.finish("Plots/.logfile.txt")
    
    ############## PLOTTING #################
    A = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)
    
    tsne =WassersteinTSNE(seed=13)
    
    embedding = tsne.fit(A)
    embedding['sizes'] = 5
    embedding.index =embedding.index.to_series(name='flags').map(labels)
    
    fig, ax = plt.subplots(figsize=(20,20))
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax)
    # fig.savefig(f'Plots/EVS_{name}_ExactWasserstein.svg')
    fig.savefig(f'Reports/Figures/EVS/ExactWasserstein.pdf')
    plt.show()
    

def comparison(name, maxnum=10000):
    
    ########### LOADING ################
    EVS = EuropeanValueStudy(max_entries=maxnum)
    labeldict = EVS.labeldict()
    dataset   = EVS.data

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(40,20))
    
    A = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)
    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A)
    embedding['sizes'] = 5
    embedding.index =embedding.index.to_series(name='flags').map(labeldict)
    
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax1)
    
    w=0.5
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = GaussianTSNE(WSDM, seed=13)
    embedding = WT.fit(w=w, angle=180)
    
    embedding.index = embedding.index.to_series(name='flags').map(labeldict)
    embedding['sizes'] = 5
    

    utils.embedFlags(embedding, title=rf"Gaussian embedding ($\lambda$={w})", ax=ax2)
 
    fig.savefig(f'Plots/EVS_{name}_WassersteinComparison.svg')
    fig.savefig(f'Reports/Figures/EVS/WassersteinComparison.pdf')
    plt.show()
    
if __name__ == '__main__':
    comparison('complete')