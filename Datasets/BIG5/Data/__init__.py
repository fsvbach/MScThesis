#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:41:02 2021

@author: fsvbach
"""

import pandas as pd

def US():
    return pd.read_csv('Datasets/BIG5/Data/BIG5-US.csv', header=0, index_col=0)

def Complete():
    return pd.read_csv('Datasets/BIG5/Data/BIG5.csv', header=0, index_col=0)

def Labels():
    label = pd.read_csv('Datasets/BIG5/Data/labels.csv', header=0, index_col=0,
                        keep_default_na=False)
    return label.continent.to_dict()