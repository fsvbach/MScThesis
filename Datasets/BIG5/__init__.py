#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:41:02 2021

@author: fsvbach
"""

import pandas as pd

def US():
    return pd.read_csv('Datasets/BIG5/BIG5-US.csv', header=0, index_col=0)

def BIG(maxnum=12000):
    data = pd.read_csv('Datasets/BIG5/BIG5.csv', header=0, index_col=0)
    datacut = []
    for c, d in data.groupby(level=0):
        if len(d) > maxnum:
            d = d.sample(maxnum, axis=0)
        datacut.append(d)
    return pd.concat(datacut)
    
def Aligned():
    data = pd.read_csv('Datasets/BIG5/BIG5.csv', header=0, index_col=0)
    info = pd.read_csv('Datasets/BIG5/labels.csv', header=0, index_col=0)
    return data.multiply(info.direction) - 3 * (info.direction - 1)
    
def Merged():
    data = Aligned()
    data.columns = [s[:3] for s in data.columns]
    return data.groupby(level=0, axis=1).mean() 
    

def Questions():
    info = pd.read_csv('Datasets/BIG5/labels.csv', header=0, index_col=0)
    return info.question.to_dict()
    
def Labels():
    label = pd.read_csv('Datasets/BIG5/continents.csv', header=0, index_col=0,
                        keep_default_na=False)
    return label.continent.to_dict()