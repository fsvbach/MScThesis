#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class Preprocess:
    _info = {'v38': ('Self-Control', 10),
                 'v39': ('Self-Satisfaction', 10), #free choice
                 'v63': ('Importance of Religion', 10),# religion
                 'v102': ('Self-Conservative', 10), # political landscape
                 'v103': ('State should provide', 10),
                 'v104': ('People can refuse work', 10),
                 'v105': ('Competition is harmful', 10),
                 'v106': ('More Incentives for effort', 10),
                 'v107': ('State should own more', 10),  # social welfare
                 'v143': ('Opinion: status quo democratic?', 10),
                 'v144': ('Opinion: system functioning?', 10),        # satisfaction with status quo
                 'v185': ('Immigration: take jobs!', 10), 
                 'v186': ('Immigration: are harmless', 10), 
                 'v187': ('Immigration: no strain on welfare', 10),
                 'v188': ('Immigrans should keep their tradtions', 10), # immigration
                 'v199': ('Wouldnt give money for environment', 5),
                 'v200': ('I can help environment', 5),
                 'v201': ('Environment very important', 5),
                 'v202': ('I alone can help environment', 5),
                 'v203': ('Climate change is not exagerated', 5) } # environment      

    marker = ['v275b_N2', 'c_abrv', 'v275b_N1']
        
    def __init__(self, countries=None, transform=False):
        self.questions = list(self._info.keys())
        self.countries = countries
        self.df        = pd.read_stata("Datasets/EVS2020/Data/EVS.dta", 
                                       convert_categoricals=False,
                                       columns = self.questions+self.marker)
        if not countries:
            self.countries = self.df.c_abrv.unique()
        
        self.df[self.df[self.questions]<0] = np.NaN
        self.df.dropna(inplace=True)
        
        for key in self.questions:
            offset = 0.05
            minval, maxval = 1, self._info[key][1]
            self.df[key] = offset + (self.df[key]-minval)*(1-2*offset)/(maxval-minval)
            
            if transform:
                self.df[key]  = np.log(self.df[key] / (1-self.df[key]))
            
    def NUTS(self, NUTS=1, min_entries=2):
        self.df.set_index(['c_abrv', f'v275b_N{NUTS}'], inplace=True, drop=True)
        data = self.df.loc[self.countries, self.questions]
        
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

if __name__ == '__main__':
    EVS = Preprocess()
    