#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:38:43 2021

@author: fsvbach
"""

from openTSNE import TSNE
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,100,size=(2000, 4)), 
                  columns=list('ABCD'))

tsne = TSNE()

# df  = df.to_numpy()

try:
    tsne.fit(df[:500])
    print('Subset of DataFrame works.')
except:
    print("Subset of DataFrame doesn't work.")
    
try:
    tsne.fit(df)
    print('Whole DataFrame works.')
except:
    print("Whole DataFrame doesn't work.")
    
tsne.fit(df)