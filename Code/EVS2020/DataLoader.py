#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class EVS:
    questions = ['v38', 'v39',          # ability to change life
              'v63',                 # religion
              'v102', 'v103', 'v104', 'v105', 'v106', 'v107',  # social welfare
              'v143', 'v144',        # satisfaction with status quo
              'v185', 'v186', 'v187', 'v188',                # immigration
              'v199', 'v201','v202','v203','v200'           # environment
              ]
        
    def __init__(self, index=['c_abrv', 'v275b_N2']):
        
        self.df = pd.read_stata("Data/EVS2020/EVS.dta", 
                           convert_categoricals=False,
                           columns = self.questions+index)
        
        self.countries = self.df.c_abrv.unique()
        
        self.df.set_index(['c_abrv', 'v275b_N2'], drop=True, inplace=True)


    def NUTS2(self, countries=None, questions=None):
        if countries:
            self.countries = countries
        if questions:
            self.questions = questions
        
        data = self.df.loc[self.countries, self.questions]
        data[data<0] = np.NaN
        
        return data.dropna()
    

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plotEVS(embedding):
    fig, ax = plt.subplots(figsize=(20,20))
    
    for country, data in embedding.groupby(level=0):
        X,Y = data['x'],data['y']
        ax.scatter(X, Y,label=country)
        flag = plt.imread(f'Data/EVS2020/flags/w640/{country.lower()}.png')
        plot_images(X, Y, flag, ax)
    
    ax.set_title('NUTS2 regions from European countries', fontsize=50)
    ax.axis('off')
    fig.savefig('Plots/tsneEVS.svg')
    plt.show()
    plt.close()
    
def plot_images(x, y, image, ax=None):
    ax = ax or plt.gca()

    for xi, yi in zip(x,y):
        im = OffsetImage(image, zoom=5/ax.figure.dpi)
        im.image.axes = ax

        ab = AnnotationBbox(im, (xi,yi), frameon=False, pad=0.0,)

        ax.add_artist(ab)
