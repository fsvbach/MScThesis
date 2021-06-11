#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:07:16 2021

@author: fsvbach
"""


import pandas as pd



def MeanStdCorr(dataset, ax=None):

    groups = dataset.groupby(level=0)
    
    a = groups.mean().stack()
    a.name = 'mean'
    b = groups.std().stack()
    b.name = 'std'
    embedding = pd.concat([a,b], axis=1)
    
    x,y = embedding.values.T
    
    ax.scatter(x,y, s=0.5)
    ax.set(xlabel='mean', ylabel='std')
    
