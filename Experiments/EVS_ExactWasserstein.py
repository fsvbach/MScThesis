#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:25:29 2021

@author: bachmafy
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Datasets.EVS2020 import EuropeanValueStudy
from WassersteinTSNE.utils import Timer
from WassersteinTSNE.Distances import WassersteinDistanceMatrix, GaussianWassersteinDistance
from WassersteinTSNE.TSNE import WassersteinTSNE, GaussianTSNE
from WassersteinTSNE.Distributions import RotationMatrix, MirrorMatrix, Dataset2Gaussians
from Experiments.Visualization import utils

name = 'complete2W'
EVS = EuropeanValueStudy(max_entries=1000)
labels  = EVS.labeldict()
dataset = EVS.data

def calculate():
    ########### CALCULATING ################
    timer = Timer('EVS Exact Wasserstein')  
    K = WassersteinDistanceMatrix(dataset, timer=timer)
    K.to_csv(f'Experiments/Distances/EVS_{name}.csv')
    # np.save(f'Datasets/EVS2020/Distances/{name}', K)  
    timer.finish("Plots/.logfile.txt")
    
def embed():
    ############## PLOTTING #################
    A = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)

    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A)
    embedding['sizes'] = 5
    embedding.index =embedding.index.to_series(name='flags').map(labels)
    
    fig, ax = plt.subplots(figsize=(20,20))
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax)
    fig.savefig(f'Plots/EVS_{name}_ExactWasserstein.svg')
    # fig.savefig(f'Reports/Figures/EVS/ExactWasserstein.pdf')
    plt.show()
    

def compareEmbeddings():
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(40,20))
    
    A = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)
    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A)
    embedding['sizes'] = 5
    embedding.index =embedding.index.to_series(name='flags').map(labels)
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax1)
    
    w=0.5
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = GaussianTSNE(WSDM, seed=13)
    embedding = WT.fit(w=w, trafo=RotationMatrix(180))
    embedding.index = embedding.index.to_series(name='flags').map(labels)
    embedding['sizes'] = 5
    utils.embedFlags(embedding, title=rf"Gaussian embedding ($\lambda$={w})", ax=ax2)
    
    fig.savefig(f'Plots/EVS_{name}_WassersteinComparison.svg')
    # fig.savefig(f'Reports/Figures/EVS/WassersteinComparison.pdf')
    plt.show()
    
def compareMatrices():
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    B = WSDM.matrix()
    
    Exact = pd.read_csv(f'Experiments/Distances/EVS_{name}.csv', index_col=0)
    A = Exact.loc[WSDM.index, WSDM.index].values

    diff = np.abs(A - B)
    perc = 100*diff/np.maximum(A,B)
    np.fill_diagonal(perc, 0)
    fig = utils.plotMatrices([A, B, diff, perc], 
                             ['Exact', 'Gaussian', 'Difference (abs)', r'Difference ($\%$)'])
    
    fig.savefig('Plots/EVS_WassersteinMatrix.svg')
    # fig.savefig(f'Reports/Figures/GER/WassersteinMatrix.pdf')
    plt.show()
    
    return np.unravel_index(perc.argmax(), perc.shape)
    
if __name__ == '__main__':
    # calculate()
    # embed()
    compareMatrices()
    # compareEmbeddings()