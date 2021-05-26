#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 13:02:33 2021

@author: fsvbach
"""

import numpy as np
import pandas as pd

class Wahlbezirke:

    head = 300
    data = pd.read_csv('Datasets/GER2017/Data/btw17_wbz_zweitstimmen.csv', delimiter=';',
                    encoding='ISO-8859-1', header=4, index_col=[1,0], usecols = [0,1,13]+list(range(17,51)))
     
    ### Creating labels
    labels  = pd.read_csv('Datasets/GER2017/Data/labels.csv',
                          index_col=0)
    
    ### Merging CDU and CSU
    data['CDU'] += data['CSU']
    data.drop('CSU', axis=1, inplace=True)
    
    ### Get rid of MultiIndex
    data = data.droplevel(0)
    
    ### Durchschnitt berechnen
    data = data.append(pd.Series(data.sum(), name=head))
    
    ### Sortieren
    data.sort_values(head, axis=1, ascending=False, inplace=True)
     
    ### In Prozent umrechnen
    data = data.divide(data['Wähler (B)'], axis='rows')*100
    data.drop('Wähler (B)', axis=1, inplace=True)
    mean = data.loc[head]
    
    ### Durchschnitt rausnehmen (und abziehen)
    data.drop(index=head, inplace=True)

    ### etwaige Nan löschen
    data.dropna(inplace=True)
    
    def __init__(self, numparty=None):
        if numparty:
            self.data = self.data.iloc[:,:numparty]
        
    def subtract_mean(self):
        self.data = (self.data-self.mean)
        return self.mean
    
if __name__ == '__main__':
    G = Wahlbezirke()
    print(np.any(G.data.isna()))