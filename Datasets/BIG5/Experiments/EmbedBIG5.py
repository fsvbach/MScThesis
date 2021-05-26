#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:17:53 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt

from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.Countries import embedFlags
from Datasets.BIG5.Data import Complete, Labels

def run(suffix=''):

    dataset = Complete()
       
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=13)
    
    fig, axes = plt.subplots(ncols=3, figsize=(45,10))
    
    for w, ax in zip([0,0.5,1], axes):
        
        embedding = WT.fit(w=w)
        embedding['sizes'] = 3
        # embedding.index = embedding.index.to_series().map(Labels())
        embedding.index = embedding.index.to_series().map(str.lower)
        embedFlags(embedding, f'embedding with w={w}', 'flags', ax=ax)
        print('Plotted subplot')
        
    fig.suptitle(f'Countries with BIG5 embedding', fontsize=30)  
    fig.savefig(f'Plots/BIG5_countries.svg')
    plt.show()
    plt.close()   



if __name__ == '__main__':
    # run(['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'BG',  'FR', 'ES'], suffix='TEST')
    run()
    