#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:02:33 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class Bundestagswahl:
    head = 300
    data = pd.read_csv('Datasets/GER2017/btw17_wbz_zweitstimmen.csv', delimiter=';',
                    encoding='ISO-8859-1', header=4, index_col=[1,0], usecols = [0,1,13]+list(range(17,51)))
         
    ### Merging CDU and CSU
    data['CDU'] += data['CSU']
    data.drop('CSU', axis=1, inplace=True)
    
    ### Get rid of MultiIndex
    data = data.droplevel(0)
    
    ### Entferne leere Wahlkreise
    data = data.loc[data['Wähler (B)'] > 20]
    
    ### Durchschnitt berechnen
    data = data.append(pd.Series(data.sum(), name=head))
    
    ### Sortieren
    data.sort_values(head, axis=1, ascending=False, inplace=True)
     
    ### In Prozent umrechnen
    size = data['Wähler (B)']
    data = data.divide(size, axis='rows')
    data.drop('Wähler (B)', axis=1, inplace=True)
    mean = data.loc[head]
    
    ### Durchschnitt rausnehmen (und abziehen)
    data.drop(index=head, inplace=True)
    size.drop(index=head, inplace=True)
    
    ### etwaige Nan löschen
    data.dropna(inplace=True)
    
    def __init__(self, numparty=None):
        if numparty:
            self.data = self.data.iloc[:,:numparty]
            self.mean = self.mean.iloc[:numparty]
        
    def subtract_mean(self):
        self.data = (self.data-self.mean)
        return self.mean
    
    def transform(self):
        data = np.arcsin(np.sqrt(self.data)).multiply(np.sqrt(self.size), axis='rows')
        return data
    
    def labeldict(self, column='Bundesland'):
        labels = pd.read_csv('Datasets/GER2017/labels.csv',
                          index_col=0)
        return labels[column].to_dict()
    
if __name__ == '__main__':
    G = Bundestagswahl()
    print(np.any(G.data.isna()))