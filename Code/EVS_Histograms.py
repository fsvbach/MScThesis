#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:26:24 2021
@author: fsvbach
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run():
    question = 'v102'
    group = 'v275b_N2'
    
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
    for i, txt in enumerate(df.c_abrv.unique()):
        plt.annotate(txt, (X[i], Y[i]))    
    plt.xlabel('mean')
    plt.ylabel('std')
    
        
    germ = df.loc[df.country == 276, question]
    germ = germ.loc[germ >= 0]
    plt.scatter(germ.mean(), germ.std(), label='Germany')
    plt.legend()
    plt.title('where are you on a scale from left=1 to right=10?')
    plt.savefig('Plots/political_landscape.svg')
    plt.show()
    plt.close()

def find(i):
    return cn.loc[cn['Numeric code'] == i, 'English short name lower case'].to_string()

def hist(i):
    alb = df.loc[df.country == i, question]
    alb = alb.loc[alb >= 0]
    alb.plot.hist()
    name = find(i)
    plt.title(name)
    plt.savefig(f'Plots/{name}.svg')
    plt.show()
    plt.close()

