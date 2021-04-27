#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:26:24 2021
@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

question = 'v144'#'v102'

df = pd.read_stata("Data/EVS.dta", convert_categoricals=False)
cn = pd.read_csv("Data/countries.csv")

countries = []
stats     = []
for code in df.country.unique():
    dim1 = df.loc[df.country == code, question]
    dim1 = dim1.loc[dim1 >= 0]
    stats.append([dim1.mean(), dim1.std()])
    countries.append(cn.loc[cn['Numeric code'] == code, 'English short name lower case'].to_string().split(" ")[-1])
    # plt.scatter(dim1.mean(), dim1.std(), label=countries[-1])
X,Y = np.array(stats).T

plt.scatter(X,Y, label='Europe')
for i, txt in enumerate(df.country.unique()):
    plt.annotate(txt, (X[i], Y[i]))    
plt.xlabel('mean')
plt.ylabel('std')

    
germ = df.loc[df.country == 276, question]
germ = germ.loc[germ >= 0]
plt.scatter(germ.mean(), germ.std(), label='Germany')
plt.legend()
plt.show()

def find(i):
    return cn.loc[cn['Numeric code'] == i, 'English short name lower case']

def hist(i):
    alb = df.loc[df.country == i, question]
    alb = alb.loc[alb >= 0]
    alb.plot.hist()

