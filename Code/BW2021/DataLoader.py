#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

import pandas as pd

class Gemeinden:
    def __init__(self, nonvoters=True):
        self.nonvoters = nonvoters

        self.head = 'Baden-Württemberg'
        
        data = pd.read_csv('Data/Election BW2021/Gemeinden.csv', delimiter=';', encoding='ISO-8859-1', index_col='Gemeinde')
        data = data.append(pd.Series(data.sum(), name=self.head))
        data = data.loc[~data.index.str.contains('Briefwahl')]

        total = data['Gültige Stimmen'] 
        if nonvoters:
            total = data['Wahlberechtigte']
            data['Nichtwähler'] = total - data['Gültige Stimmen'] 

        data.drop(columns=data.columns[:8], inplace=True)
        data.fillna(0, inplace=True)
        data.sort_values(self.head, axis=1, ascending=False, inplace=True)
         
        data = data.divide(total, axis='rows')*100
        self.average = data.loc[self.head]
        
        data.drop(index=self.head, inplace=True)
        self.data = (data-self.average)

    def title(self):
        title = f'Gemeinden of {self.head}'
        if self.nonvoters:
            title += ' with Non-Voters'
        return title
    
    def colors(self):
        colors = pd.read_csv('Data/Election BW2021/Farben.csv', delimiter=';', encoding='utf-8', header=0)
        colors = colors.iloc[0].to_dict()
        if self.nonvoters:
            colors['Nichtwähler'] = 'grey'
        return colors
        

class Germany:
    data = pd.read_csv('Data/Election GER2017/GER Wahlkreise.csv', delimiter=';',
                   encoding='utf-8', header=[5,6,7])
    
    
class Wahlkreise:
    def __init__(self, nonvoters=True):
        self.nonvoters = nonvoters
        self.head = 'Land Baden-Württemberg'
        
        data = data = pd.read_csv('Data/Election BW2021/Kreise.csv', delimiter=';', encoding='ISO-8859-1', index_col='Wahlkreis')

        total = data['Gültige Stimmen'] 
        if nonvoters:
            total = data['Wahlberechtigte']
            data['Nichtwähler'] = total - data['Gültige Stimmen'] 

        data.drop(columns=data.columns[:8], inplace=True)
        data.fillna(0, inplace=True)
        data.sort_values(self.head, axis=1, ascending=False, inplace=True)
         
        data = data.divide(total, axis='rows')*100
        self.average = data.loc[self.head]
        
        data.drop(index=self.head, inplace=True)
        self.data = (data-self.average)

    def title(self):
        title = f'Wahlkreise of {self.head}'
        if self.nonvoters:
            title += ' with Non-Voters'
        return title
    
    def colors(self):
        colors = pd.read_csv('Data/Election BW2021/Farben.csv', delimiter=';', encoding='utf-8', header=0)
        colors = colors.iloc[0].to_dict()
        if self.nonvoters:
            colors['Nichtwähler'] = 'grey'
        return colors
        