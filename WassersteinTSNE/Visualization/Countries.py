#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:47:11 2021

@author: bachmafy
"""

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt

def embedFlags(embedding, title, folder, ax=None):
    if not ax:
        ax = plt.gca()
        
    for label, data in embedding.groupby(level=0):
        X, Y, s = data['x'], data['y'], data['sizes']
        ax.scatter(X, Y,label=label, s=s/100)
        flag = plt.imread(f'MISC/Images/{folder}/{label}.png')
        plotImages(X, Y, flag, s, ax)

    ax.set(title=title)
    ax.axis('off')

   
def plotImages(x, y, image, sizes, ax=None):
    ax = ax or plt.gca()

    for xi, yi, zm in zip(x,y, sizes):
        im = OffsetImage(image, zoom=zm/ax.figure.dpi)
        im.image.axes = ax

        ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)

        ax.add_artist(ab)