#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:13:59 2021

@author: bachmafy
"""

PATH = 'Experiments/Visualization/Images'

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse
import matplotlib
import numpy as np
import pandas as pd

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 12,
    'text.usetex': True,
    'pgf.rcfonts': False})

def plotMatrix(matrices, titles, name):
    n = len(matrices)
    fig, axes = plt.subplots(ncols=n, figsize=(7*n,5))
    for matrix, title, ax in zip(matrices, titles, axes):
        m = ax.imshow(matrix, cmap='Greens')
        ax.set(title=title)
        plt.colorbar(m, ax=ax)
    fig.savefig(f'Plots/{name}.svg')
    plt.show()
    plt.close()

   
def plotGaussian(Gaussian, size=20, STDS=[1,2,3], color='black', ax=None):
    if not ax:
        ax = plt.gca()
        
    for i in STDS:
        mean, width, height, angle = Gaussian.shape(std=i)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor=color, facecolor='none', 
                      linewidth=2, linestyle='--')
        ax.add_patch(ell)
        
    if size:
        samples = Gaussian.samples(size)
        x,y = samples.T
        ax.scatter(x,y)
        
    return ell


def embedFlags(embedding, title, ax=None):
    if not ax:
        ax = plt.gca()
        
    for label, data in embedding.groupby(level=0):
        X, Y, s = data['x'], data['y'], data['sizes']
        ax.scatter(X, Y,label=label, s=s/100)
        flag = plt.imread(f'{PATH}/{embedding.index.name}/{label}.png')
        plotImages(X, Y, flag, s, ax)

    ax.set_title(title, fontsize=50)
    ax.axis('off')

   
def plotImages(x, y, image, sizes, ax=None):
    ax = ax or plt.gca()

    for xi, yi, zm in zip(x,y, sizes):
        im = OffsetImage(image, zoom=zm/ax.figure.dpi)
        im.image.axes = ax

        ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)

        ax.add_artist(ab)

def MeanStdCorr(dataset, title='', ax=None):
    groups = dataset.groupby(level=0)
    a = groups.mean().stack()
    a.name = 'mean'
    b = groups.std().stack()
    b.name = 'std'
    embedding = pd.concat([a,b], axis=1)
    x,y = embedding.values.T
    ax.scatter(x,y, s=0.5)
    ax.set(xlabel='mean', ylabel='std', title=title)
    
    
def get_rectangle(N):
    A = int(np.sqrt(N))
    B = int(N/A) + (N%A>0) 
    return A,B


def border(ax, color):
    ax.axis('on')
    ax.set_xticks([], minor=[])
    ax.set_yticks([], minor=[])
    for spine in ['bottom', 'top', 'left', 'right']:
        ax.spines[spine].set_color(color)

def code2name():
    labels = pd.read_csv('Experiments/Visualization/Images/countries.csv',
                index_col=1)
    return labels['English short name lower case'].to_dict()
    