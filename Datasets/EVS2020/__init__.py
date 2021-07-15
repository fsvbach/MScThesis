#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class EuropeanValueStudy:
    
    UP   = (1,10)
    DOWN = (10,1)
    up   = (1, 5)
    down = (5, 1)
    
    overview =  {'v38': ('I have complete control over my life', UP),
                 'v39': ('I am satisfied with my life', UP), #free choice
                 'v63': (' God is important in my life', UP),# religion
                 'v102': ("I consider myself 'on the right'", UP), # political landscape
                 'v103': ('Everyone is responible for him/herself', DOWN),
                 'v104': ('The unemployed should take ANY job', DOWN),
                 'v105': ('Competition is good', DOWN),
                 'v106': ('Incomes should be made equal', DOWN),
                 'v107': ('Private ownership should be increased', DOWN),  # social welfare
                 'v143': ('My country is governed democratically', UP),
                 'v144': ('I am satisfied with the government', UP),        # satisfaction with status quo
                 'v185': ("'Immigrants take jobs away!'", DOWN), 
                 'v186': ("'Immigrants make crime problems'", DOWN), 
                 'v187': ('Immigration is a strain on welfare system', DOWN),
                 'v188': ('Immigrans should maintain their tradtions', DOWN), # immigration
                 'v199': ('I would give money for environment', down),
                 'v200': ("Someone like me can do much for environment", up),
                 'v201': ('There are more important things than environment', down),
                 'v202': ('Others should start to protect the environment', down),
                 'v203': ('Environmental threats are exaggerated', down),
                'v149':	('do you justify: claiming state benefits', UP),
                'v159':	('do you justify: avoiding a fare on public transport', UP),
                'v150':	('do you justify: cheating on tax', UP),
                'v152':	('do you justify: accepting a bribe', UP),
                'v153':	('do you justify: homosexuality', UP),
                'v160':	('do you justify: prostitution', UP),
                'v154':	('do you justify: abortion', UP),
                'v155':	('do you justify: divorce', UP),
                'v156':	('do you justify: euthanasia', UP),
                'v157':	('do you justify: suicide', UP),
                'v158': ('do you justify: having casual sex', UP),
                'v163':	('do you justify: death penalty', UP)}

    marker    = ['v275b_N2', 'c_abrv', 'v275b_N1']

    def __init__(self, min_entries=40, max_entries=4000):
        

        questions = list(self.overview.keys())
            
        df        = pd.read_stata("Datasets/EVS2020/EVS.dta", 
                                   convert_categoricals=False,
                                   columns = questions+self.marker)
        
        print(f"Loaded {len(df)} questionaires")
        df[df[questions]<0] = np.NaN
        df.dropna(inplace=True)
        print(f"Kept {len(df)} non-empty questionaires")
    
        ### germany has different structure
        de = df.loc[df.c_abrv == 'DE']
        de.set_index('v275b_N1', inplace=True, drop=True)
        de = de.drop('v275b_N2', axis=1)
          
        ### merging germany with rest      
        df.set_index('v275b_N2', inplace=True, drop=True)
        df.drop(['-4','-1'], inplace=True)
        df.drop('v275b_N1', axis=1, inplace=True)
        
        ### labels
        data = pd.concat([de,df])
        self.labels = data.c_abrv.map(str.lower).to_dict()
        data.drop('c_abrv', axis=1, inplace=True)
        
        sizes = data.groupby(level=0).size() 
    
        self.data = data.loc[ (min_entries < sizes) & (sizes < max_entries)]
    
    def labeldict(self):
        return self.labels 
        
    # if not countries:
    #     countries = df.c_abrv.unique()
    
    # for key in questions:
    #     offset = 0.05
    #     minval, maxval = overview[key][1]
    #     df[key] = offset + (df[key]-minval)*(1-2*offset)/(maxval-minval)
        
    #     if transform:
    #         df[key]  = np.log(df[key] / (1-df[key]))
      
    
if __name__ == '__main__':
    EVS = EuropeanValueStudy()
    data = EVS.data
    # labels = EVS.labels