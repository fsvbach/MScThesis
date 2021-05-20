#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:47:11 2021

@author: bachmafy
"""

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt

def plotEVS(embedding, name, size=10, ax=None):
    if not ax:
        ax = plt.gca()
        
    for country, data in embedding.groupby(level=0):
        X,Y = data['x'],data['y']
        ax.scatter(X, Y,label=country)
        flag = plt.imread(f'Datasets/EVS2020/Data/flags/w640/{country.lower()}.png')
        for xi, yi, size in zip(X,Y,data['sizes']):
            im = OffsetImage(flag, zoom=size/ax.figure.dpi)
            im.image.axes = ax
            ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)
            ax.add_artist(ab)
    
    ax.set_title(f'{name}', fontsize=30)
    ax.axis('off')

    
