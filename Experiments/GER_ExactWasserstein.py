import warnings
warnings.filterwarnings("ignore")

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
from WassersteinTSNE.Distances import WassersteinDistanceMatrix, GaussianWassersteinDistance
from WassersteinTSNE.TSNE import WassersteinTSNE, GaussianTSNE
from WassersteinTSNE.Distributions import RotationMatrix, MirrorMatrix, Dataset2Gaussians
from Experiments.Visualization import utils


GER = Bundestagswahl()
labels  = GER.labeldict()
dataset = GER.data

def calculate():
    ########### CALCULATING ################
    timer = Timer('GER Exact Wasserstein')
    K = WassersteinDistanceMatrix(dataset, timer=timer)
    K.to_csv('Experiments/Distances/GER_ExactWasserstein.csv')
    timer.finish("Plots/.logfile.txt")
    
def embed():
    ############## PLOTTING #################
    A = pd.read_csv('Experiments/Distances/GER_ExactWasserstein_A.csv', index_col=0)
    
    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A)
    embedding['sizes'] = 20
    embedding.index =embedding.index.to_series(name='wappen').map(labels)
    
    fig, ax = plt.subplots(figsize=(20,20))
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax)
    fig.savefig('Plots/GER_ExactWasserstein.svg')
    # fig.savefig(f'Reports/Figures/GER/ExactWasserstein.pdf')
    plt.show()
    

def compareEmbeddings():
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(50,25))
    
    A = pd.read_csv('Experiments/Distances/GER_ExactWasserstein_A.csv', index_col=0)
    tsne =WassersteinTSNE(seed=13)
    embedding = tsne.fit(A, trafo=MirrorMatrix([1,0])@RotationMatrix(90))
    embedding['sizes'] = 20
    embedding.index =embedding.index.to_series(name='wappen').map(labels)
    utils.embedFlags(embedding, 'Exact Wasserstein embedding', ax=ax1)
    
    w=0.5
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = GaussianTSNE(WSDM, seed=13)
    embedding = WT.fit(w=w, trafo=MirrorMatrix([1,0])@RotationMatrix(90))
    embedding.index = embedding.index.to_series(name='wappen').map(labels)
    embedding['sizes'] = 20
    utils.embedFlags(embedding, title=rf"Gaussian embedding ($\lambda$={w})", ax=ax2)
 
    fig.savefig('Plots/GER_WassersteinComparison.svg')
    fig.savefig('Reports/Figures/GER/WassersteinComparison.pdf')
    plt.show()

def compareMatrices():
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    B = WSDM.matrix()
    
    Exact = pd.read_csv('Experiments/Distances/GER_ExactWasserstein_A.csv', index_col=0)
    A = Exact.loc[WSDM.index, WSDM.index].values

    fig = utils.plotMatrices([A, B, np.abs(A - B)], 
                             ['Exact', 'Gaussian', r'Difference'])
    
    fig.savefig('Plots/GER_WassersteinMatrix.svg')
    # fig.savefig(f'Reports/Figures/GER/WassersteinMatrix.pdf')
    plt.show()
    
    
if __name__ == '__main__':
    # calculate()
    # embed()
    compareEmbeddings()