#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:47:11 2021

@author: bachmafy
"""

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt

def plotEVS(embedding, title, ax=None):
    if not ax:
        ax = plt.gca()
        
    for country, data in embedding.groupby(level=0):
        X,Y = data['x'],data['y']
        ax.scatter(X, Y,label=country)
        flag = plt.imread(f'Datasets/EVS2020/Data/flags/w640/{country.lower()}.png')
        plotImages(X, Y, flag, data['sizes'], ax)

    ax.set(title=title)
    ax.axis('off')
    
def plotGER(embedding, title, ax=None):
    if not ax:
        ax = plt.gca()
        
    for land, data in embedding.groupby(level=0):
        X,Y = data['x'],data['y']
        ax.scatter(X, Y,label=land)
        flag = plt.imread(f'Datasets/GER2017/Data/flags/{land}.png')
        plotImages(X, Y, flag, data['sizes'], ax)

    ax.set(title=title)
    ax.axis('off')
   
def plotImages(x, y, image, sizes, ax=None):
    ax = ax or plt.gca()

    for xi, yi, zm in zip(x,y, sizes):
        im = OffsetImage(image, zoom=zm/ax.figure.dpi)
        im.image.axes = ax

        ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)

        ax.add_artist(ab)