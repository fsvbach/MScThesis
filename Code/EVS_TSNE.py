#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""


from .share.Distances import WassersteinTSNE
from .DataLoader.EVS2020 import EVS
from .Styles.EVS import plotEVS

import pandas as pd
import numpy as np


def run():
    dataset = EVS()
    
    countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'ES', 'BG',  'FR']
    
    labels = []
    NUTS2_means  = []
    NUTS2_covs   = []
    NUTS2_size = []
    
    for i, country in dataset.NUTS2().groupby(level=0):
        for j, nuts in country.groupby(level=1):
            NUTS2_means.append(nuts.mean())
            # NUTS2_covs.append(nuts.cov())
            labels.append(i)
    
    data = pd.concat(NUTS2_means, axis=1).T
    
    tsne = WassersteinTSNE(seed=13)
    
    embedding = tsne.fit(data.to_numpy())
    
    
    embedding = pd.DataFrame(embedding, index=labels, columns=['x','y'])
    
    
    plotEVS(embedding)
        
if __name__ == '__main__':
    run()