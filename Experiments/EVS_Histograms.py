#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:26:24 2021
@author: fsvbach
"""

from Datasets import EVS2020 as Data
from Experiments.Visualization.utils import code2name
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

fullnames = code2name()

feature = 'v102'
selection=['AL', 'GB', 'DE', 'NL']
NUTS=1

normal, labels = Data.LoadEVS(Data.overview,
                               countries=selection, 
                               transform=False, 
                               NUTS=NUTS, 
                               min_entries=40) 
trafo, _ = Data.LoadEVS(Data.overview,
                               countries=selection, 
                               transform=True, 
                               NUTS=NUTS, 
                               min_entries=40) 
countries = normal.index.to_series().map(labels)
trafo.index  = countries.map(str.upper)
normal.index = countries.map(str.upper)

for country in selection:
    fig, axes = plt.subplots(1,2, figsize=(8,4))

    for dataset, ax, xlabel in zip([normal, trafo], axes, ['Score', 'Logits']):
        data = dataset.loc[country, feature]
        
        # the histogram of the data
        _, bins, _ = ax.hist(data, 18, density=1, alpha=0.5)
        
        # add a 'best fit' line
        mu, sigma = norm.fit(data)
        best_fit_line = norm.pdf(bins, mu, sigma)
        ax.plot(bins, best_fit_line)
        
        #plot
        ax.set(xlabel = xlabel,
                ylabel= 'Probability Density',
                title = f'{fullnames[country]}: '+r'$\mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

    fig.tight_layout()
    fig.savefig(f'Plots/EVS_HistoTrafo_{country}.pdf')
    fig.savefig(f'Reports/Figures/EVS/HistoTrafo_{country}.pdf')
    plt.show()
    plt.close()