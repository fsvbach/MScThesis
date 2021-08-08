#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:46:02 2021

@author: fsvbach
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Datasets.GER2017 import Bundestagswahl
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import WassersteinDistanceMatrix
from WassersteinTSNE.TSNE import WassersteinTSNE
from Experiments.Visualization import utils

from WassersteinTSNE import Dataset2Gaussians, GaussianTSNE, GaussianWassersteinDistance, _naming, RotationMatrix

GER = Bundestagswahl()
labels  = GER.labeldict()
dataset = GER.data

def calculate():
    ########### CALCULATING ################
    timer = Timer('GER Exact Wasserstein')
    K = WassersteinDistanceMatrix(dataset, timer=timer)
    K.to_csv(f'Experiments/Distances/GER_ExactWasserstein.csv')
    timer.finish("Plots/.logfile.txt")
    
def embed():
    ############## PLOTTING #################
    A = pd.read_csv(f'Experiments/Distances/GER_ExactWasserstein.csv', index_col=0)
    
    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A)
    embedding['sizes'] = 20
    embedding.index =embedding.index.to_series(name='wappen').map(labels)
    
    fig, ax = plt.subplots(figsize=(20,20))
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax)
    fig.savefig(f'Plots/GER_ExactWasserstein.svg')
    # fig.savefig(f'Reports/Figures/GER/ExactWasserstein.pdf')
    plt.show()
    

def compare():
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(40,20))
    
    A = pd.read_csv(f'Experiments/Distances/GER_ExactWasserstein.csv', index_col=0)
    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A)
    embedding['sizes'] = 20
    embedding.index =embedding.index.to_series(name='wappen').map(labels)
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax1)
    
    w=0.75
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = GaussianTSNE(WSDM, seed=13)
    embedding = WT.fit(w=w, angle=180)
    embedding.index = embedding.index.to_series(name='wappen').map(labels)
    embedding['sizes'] = 20
    utils.embedFlags(embedding, title=rf"Gaussian embedding ($\lambda$={w})", ax=ax2)
 
    fig.savefig(f'Plots/GER_WassersteinComparison.svg')
    fig.savefig(f'Reports/Figures/GER/WassersteinComparison.pdf')
    plt.show()
    
if __name__ == '__main__':
    calculate()
    embed()