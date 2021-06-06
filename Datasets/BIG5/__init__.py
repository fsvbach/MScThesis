#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:41:02 2021

@author: fsvbach
"""

import pandas as pd

def US():
    return pd.read_csv('Datasets/BIG5/BIG5-US.csv', header=0, index_col=0)

def Complete():
    return pd.read_csv('Datasets/BIG5/BIG5.csv', header=0, index_col=0)

def Aligned():
    data = pd.read_csv('Datasets/BIG5/BIG5.csv', header=0, index_col=0)
    info = pd.read_csv('Datasets/BIG5/labels.csv', header=0, index_col=0)
    return data.multiply(info.direction)
    
def Merged():
    pass

def Questions():
    info = pd.read_csv('Datasets/BIG5/labels.csv', header=0, index_col=0)
    return info.question.to_dict()
    
def Labels():
    label = pd.read_csv('Datasets/BIG5/continents.csv', header=0, index_col=0,
                        keep_default_na=False)
    return label.continent.to_dict()