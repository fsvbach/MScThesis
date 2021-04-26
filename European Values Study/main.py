#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:26:24 2021
@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_stata("Data/EVS_all.dta", convert_categoricals=False)
cn = pd.read_csv("Data/countries.csv")

countries = []
stats     = []
for code in df.cntry.unique():
    dim1 = df.loc[df.cntry == code, 'E033']
    dim1 = dim1.loc[dim1 >= 0]
    stats.append([dim1.mean(), dim1.std()])
    countries.append(cn.loc[cn['Numeric code'] == code, 'English short name lower case'].to_string().split(" ")[-1])
    # plt.scatter(dim1.mean(), dim1.std(), label=countries[-1])
X,Y = np.array(stats).T

plt.scatter(X,Y)
for i, txt in enumerate(df.cntry.unique()):
    plt.annotate(txt, (X[i], Y[i]))    
plt.xlabel('mean')
plt.ylabel('std')
plt.legend()
    
germ = df.loc[df.cntry == 276, 'E033']
germ = germ.loc[germ >= 0]
plt.scatter(germ.mean(), germ.std(), label='germany')
plt.show()

def find(X):
    return cn.loc[cn['Numeric code'] == X, 'English short name lower case']

alb = df.loc[df.cntry == 8, 'E033']
alb = alb.loc[alb >= 0]
alb.plot.hist()