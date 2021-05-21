#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:10:25 2021

@author: fsvbach
"""

import pandas as pd
import matplotlib.pyplot as plt

class plotElection:
    params = {'size'     : 15,
              'numparty' : 6,
              'xstretch' : 30,
              'ystretch' : 50,
              'barwidth' : 20,
              'barheight': 3,
              'legend'   : (3, 0, 0)}
    
    def __init__(self, dataset, embedding, total):
        self.dataset   = dataset
        self.embedding = embedding
        self.total     = total

        ### Loading colors
        colors = pd.read_csv('WassersteinTSNE/Visualization/share/Parteifarben.csv', delimiter=';', encoding='utf-8', header=0)
        self.colors = colors.iloc[0].to_dict()

    def show(self, ax, **kwargs):
        self.params.update(kwargs)
        
        X, Y = self.embedding['x'], self.embedding['y']


        dist = self.params['barwidth']/self.params['numparty']
        for i in range(self.params['numparty']):
            party = self.dataset.columns[i]
            color = self.colors[party]
            offset = (self.params['numparty']/2-i)*dist-dist/2
            ax.bar(self.params['xstretch']*X-offset, 
                   self.params['barheight']*self.dataset[party], 
                   bottom=self.params['ystretch']*Y, 
                   width=dist, 
                   color=color, 
                   align='center')
            
            ### Plot Legend at xy
            l, xloc, yloc = self.params['legend']
            ax.bar(xloc + self.params['xstretch']*min(X)+l*dist*i, 
                   self.total[i]*self.params['barheight'], 
                   bottom=self.params['ystretch']*min(Y) + yloc,
                   color=color, 
                   width=l*dist, 
                   align='edge')
        
        for name, x, y in zip(self.dataset.index, X, Y):
            # text = ' '.join(name.split(' ')[1:])
            text = name
            sub = self.dataset.loc[name].max()
            ax.annotate(text, 
                        (self.params['xstretch']*x-dist/2, 
                         self.params['ystretch']*y+sub*self.params['barheight']), 
                        ha='center')
        
        ax.axis('off')

    
# def plotWahlkreis(district):
#     name = ' '.join(data.loc[district].name.split(' ')[1:])
#     fig, ax = plt.subplots()
#     ax.bar(data.columns, data.loc[district])
#     ax.set(ylabel='%',
#            title=name)
#     ax.tick_params(axis='x', labelrotation = 90)
#     fig.savefig(f'Plots/Wahlkreise/{district}.png')
#     plt.show()
#     plt.close()

# def plotBar(values, name):
#     fig, ax = plt.subplots()
#     ax.bar(data.columns, values)
#     ax.set(ylabel='%',
#            title=name)
#     ax.tick_params(axis='x', labelrotation = 90)
#     fig.savefig(f'Plots/Wahlkreise/{name}.png')
#     plt.show()
#     plt.close()