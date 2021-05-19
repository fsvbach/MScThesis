#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class Preprocess:
    questions = ['v38', 'v39',          # ability to change life
              'v63',                 # religion
              'v102', 'v103', 'v104', 'v105', 'v106', 'v107',  # social welfare
              'v143', 'v144',        # satisfaction with status quo
              'v185', 'v186', 'v187', 'v188',                # immigration
              'v199', 'v201','v202','v203','v200'           # environment
              ]
    
    marker = ['v275b_N2', 'c_abrv']
        
    def __init__(self):
        self.df        = pd.read_stata("Datasets/EVS2020/Data/EVS.dta", 
                                       convert_categoricals=False,
                                       columns = self.questions+self.marker)
        self.countries = self.df.c_abrv.unique()

    def NUTS2(self, countries=None, questions=None, min_entries=2):
        if not countries:
            countries = self.countries
        if not questions:
            questions = self.questions
        
        self.df.set_index(['c_abrv', 'v275b_N2'], inplace=True, drop=True)
        data = self.df.loc[countries, questions]
        
        data[data<0] = np.NaN
        data.dropna(inplace=True)
        
        ### labels
        labels  = []
        dataset = []
        for n, nuts in data.groupby(level=1):
            country = nuts.index.get_level_values(0).unique()
            if len(country) > 1:
                print("Labels not unique!", nuts)
                # raise AssertionError
            elif len(nuts) >= min_entries:
                labels.append(country[0])
                dataset.append(nuts.droplevel(0))

        return pd.concat(dataset), labels
        