#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:02:33 2021

@author: fsvbach
"""

import pandas as pd

class Germany:
    data = pd.read_csv('Datasets/GER2017/Data/GER Wahlkreise.csv', delimiter=';',
                   encoding='utf-8', header=[5,6,7])
    
    