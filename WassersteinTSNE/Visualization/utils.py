#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:13:59 2021

@author: bachmafy
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plotMatrix(matrices, titles, name):
    n = len(matrices)
    fig, axes = plt.subplots(ncols=n, figsize=(7*n,5))
    for matrix, title, ax in zip(matrices, titles, axes):
        m = ax.imshow(matrix)
        ax.set(title=title)
        plt.colorbar(m, ax=ax)
    # fig.savefig(f'Plots/{name}.svg')
    plt.show()
    plt.close()

   
def plotGaussians(Gaussian, size=20):
    fig, ax = plt.subplots()
    for i in range(1,4):
        mean, width, height, angle = Gaussian.shape(std=i)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor='black', facecolor='none', 
                      linewidth=2, linestyle='--')
        ax.add_patch(ell)
    samples = Gaussian.samples(size)
    x,y = samples.T
    ax.scatter(x,y)
    # plt.title(name)
    # plt.savefig(f'Plots/{name}.svg')
    plt.show()
    plt.close()


