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
        
    def __init__(self, index=['c_abrv', 'v275b_N2']):
        
        self.df = pd.read_stata("Datasets/EVS2020/Data/EVS.dta", 
                           convert_categoricals=False,
                           columns = self.questions+index)
        
        self.countries = self.df.c_abrv.unique()
        
        self.df.set_index(['c_abrv', 'v275b_N2'], drop=True, inplace=True)


    def NUTS2(self, countries=None, questions=None):
        if countries:
            self.countries = countries
        if questions:
            self.questions = questions
        
        data = self.df.loc[self.countries, self.questions]
        data[data<0] = np.NaN
        
        return data.dropna()
    


