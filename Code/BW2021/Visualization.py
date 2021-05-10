#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:10:25 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt

class plotElection:
    params = {'size'     : 15,
              'numparty' : 6,
              'xstretch' : 30,
              'ystretch' : 50,
              'barwidth' : 20,
              'barheight': 3,
              'legend'   : 3}
    
    def __init__(self, dataset):
        self.dataset = dataset

    def tSNE(self, embedding, **kwargs):
        self.params.update(kwargs)
        
        X,Y    = embedding.T
        data   = self.dataset.data
        colors = self.dataset.colors()
        title  = self.dataset.title()
        
        fig, ax = plt.subplots(figsize=(self.params['size'],
                                        self.params['size']))
        
        dist = self.params['barwidth']/self.params['numparty']
        for i in range(self.params['numparty']):
            party = data.columns[i]
            color = colors[party]
            offset = (self.params['numparty']/2-i)*dist-dist/2
            ax.bar(self.params['xstretch']*X-offset, 
                   self.params['barheight']*data[party], 
                   bottom=self.params['ystretch']*Y, 
                   width=dist, 
                   color=color, 
                   align='center')
            
            ax.bar(self.params['xstretch']*min(X)+self.params['legend']*dist*i, 
                   self.dataset.average[i]*self.params['barheight'], 
                   bottom=self.params['ystretch']*min(Y),
                   color=color, 
                   width=self.params['legend']*dist, 
                   align='edge')
        
        for name, x, y in zip(data.index, X, Y):
            text = ' '.join(name.split(' ')[1:])
            sub = data.loc[name].max()
            ax.annotate(text, 
                        (self.params['xstretch']*x-dist/2, 
                         self.params['ystretch']*y+sub*self.params['barheight']), 
                        ha='center')
            
        
        # for spine in ax.spines:
        #     ax.spines[spine].set(visible=False)
        ax.axis('off')
        # ax.set(xticks=[],
        #        yticks=[])
        ax.set_title(title, fontdict={'fontsize': 25})
        fig.savefig(f'Plots/{title}.svg')
        plt.show()
    
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