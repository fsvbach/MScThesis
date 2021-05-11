#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

from Code.share.Simulations import CovarianceMatrix, GaussianDistribution
from Code.share.Wasserstein import WassersteinTSNE, GaussianWassersteinDistance
from Code.EVS2020 import DataLoader, Style

import pandas as pd
import numpy as np


dataset = DataLoader.EVS()

countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'BG',  'FR']

labels = []
GaussianNUTS2  = []

for i, country in dataset.NUTS2().groupby(level=0):
    for j, nuts in country.groupby(level=1):
        size=len(nuts)
        if size > 40:
            mean = np.mean(nuts, axis=0)
            cov  = (nuts-mean).T@(nuts-mean)/(size-1)
            cov  = CovarianceMatrix(cov, from_array=True)
            GaussianNUTS2.append(GaussianDistribution(mean, cov))
            labels.append(i)

tsne = WassersteinTSNE(seed=13)

WSDM = GaussianWassersteinDistance(GaussianNUTS2)

embedding = tsne.fit_precomputed(WSDM.matrix(w=1))

embedding = pd.DataFrame(embedding, index=labels, columns=['x','y'])


Style.plotEVS(embedding)
        
