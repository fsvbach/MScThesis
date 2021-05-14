#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:47:11 2021

@author: bachmafy
"""

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt

def plotEVS(embedding, name, ax=None):
    if not ax:
        ax = plt.gca()
        
    for country, data in embedding.groupby(level=0):
        X,Y = data['x'],data['y']
        # ax.scatter(X, Y,label=country)
        flag = plt.imread(f'Data/EVS2020/flags/w640/{country.lower()}.png')
        plot_images(X, Y, flag, ax)
    
    ax.set_title(f'{name}', fontsize=10)
    ax.axis('off')

    
def plot_images(x, y, image, ax=None):
    ax = ax or plt.gca()

    for xi, yi in zip(x,y):
        im = OffsetImage(image, zoom=10/ax.figure.dpi)
        im.image.axes = ax

        ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)

        ax.add_artist(ab)