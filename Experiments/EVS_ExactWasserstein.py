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


def calculate(name, maxnum=10000):
    
    ########### CALCULATING ################
    timer = Timer('EVS Exact Wasserstein')
    
    EVS = EuropeanValueStudy(max_entries=maxnum)
    labels  = EVS.labeldict()
    dataset = EVS.data
    
    K = WassersteinDistanceMatrix(dataset, timer=timer)
    K.to_csv(f'Datasets/EVS2020/Distances/{name}.csv')
    np.save(f'Datasets/EVS2020/Distances/{name}', K)  
    
    timer.finish("Plots/.logfile.txt")
    
    ############## PLOTTING #################
    A = pd.read_csv(f'Datasets/EVS2020/Distances/{name}.csv', index_col=0)
    
    tsne =WassersteinTSNE(seed=13)
    
    embedding = tsne.fit(A)
    embedding['sizes'] = 5
    embedding.index =embedding.index.to_series(name='flags').map(labels)
    
    fig, ax = plt.subplots(figsize=(20,20))
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax)
    fig.savefig(f'Plots/EVS_{name}_ExactWasserstein.svg')
    plt.show()
    
if __name__ == '__main__':
    calculate('test')