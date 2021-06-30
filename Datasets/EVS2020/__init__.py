#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd


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
             'v203': ('Environmental threats are exaggerated', down) } # environment      


small    =  {'v38': ('I have complete control over my life', UP),
             # 'v39': ('I am satisfied with my life', UP), #free choice
             'v63': (' God is important in my life', UP),# religion
             'v102': ("I consider myself 'on the right'", UP), # political landscape
             # 'v103': ('Everyone is responible for him/herself', DOWN),
             # 'v104': ('The unemployed should take ANY job', DOWN),
             # 'v105': ('Competition is good', DOWN),
             # 'v106': ('Incomes should be made equal', DOWN),
             # 'v107': ('Private ownership should be increased', DOWN),  # social welfare
             # 'v143': ('My country is governed democratically', UP),
             'v144': ('I am satisfied with the government', UP),        # satisfaction with status quo
             # 'v185': ("'Immigrants take jobs away!'", DOWN), 
             'v186': ("'Immigrants make crime problems'", DOWN), 
             # 'v187': ('Immigration is a strain on welfare system', DOWN),
             # 'v188': ('Immigrans should maintain their tradtions', DOWN), # immigration
             # 'v199': ('I would give money for environment', down),
             # 'v200': ("Someone like me can do much for environment", up),
             'v201': ('There are more important things than environment', down),
             # 'v202': ('Others should start to protect the environment', down),
             # 'v203': ('Environmental threats are exaggerated', down) } # environment      
             }
# democracy = {'v133': 'Governments tax the rich and subsidize the poor'
# Religious authorities ultimately
# interpret the laws
# People choose their leaders in
# free elections
# People receive state aid for
# unemployment
# The army takes over when
# government is incompetent
# Civil rights protect people from
# state oppression
# The state makes people’s
# incomes equal
# People obey their rulers.
# Women have the same rights as
# men 

# +++ v153 MORALS

marker    = ['v275b_N2', 'c_abrv', 'v275b_N1']


def LoadEVS(topic, countries=None, transform=False, NUTS=1, min_entries=2):

    questions = list(topic.keys())
        
    df        = pd.read_stata("Datasets/EVS2020/EVS.dta", 
                               convert_categoricals=False,
                               columns = questions+marker)
    df[df[questions]<0] = np.NaN
    df.dropna(inplace=True)

    if not countries:
        countries = df.c_abrv.unique()
    
    for key in questions:
        offset = 0.05
        minval, maxval = overview[key][1]
        df[key] = offset + (df[key]-minval)*(1-2*offset)/(maxval-minval)
        
        if transform:
            df[key]  = np.log(df[key] / (1-df[key]))
       
    df.set_index(['c_abrv', f'v275b_N{NUTS}'], inplace=True, drop=True)
    data = df.loc[countries, questions]
    
    ### labels
    labels = {}
    dataset = []
    for n, nuts in data.groupby(level=1):
        country = nuts.index.get_level_values(0).unique()
        if len(country) > 1:
            print("Labels not unique!", nuts)
            # raise AssertionError
        elif len(nuts) >= min_entries:
            labels[n] = country[0].lower()
            dataset.append(nuts.droplevel(0))

    return pd.concat(dataset), labels

