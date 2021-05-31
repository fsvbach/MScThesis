#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:39:49 2021

@author: fsvbach
"""


from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from WassersteinTSNE.Visualization.utils import embedFlags

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

config = {'folder': None,
           'dataset': None,
           'name': None,
           'w': 1,
           'size': (5,25),
           'seed': None,
           'suffix': '',
           'renaming': lambda name: name}

def WassersteinEmbedding(dataset, labeldict, **kwargs):
    config.update(kwargs)
    
    fig, axes = plt.subplots(1,3, figsize=(45,15))
        
    w = 0
    for ax in axes.T.flatten():
        Gaussians = Dataset2Gaussians(dataset)
        WSDM = GaussianWassersteinDistance(Gaussians)
        WT = WassersteinTSNE(WSDM, seed=config['seed'])
        embedding = WT.fit(w=w)
        embedding.index = embedding.index.to_series(name=config['folder']).map(labeldict)
        embedding['sizes'] = np.mean(config['size'])
        embedFlags(embedding, title=f"embedding (w={w})", ax=ax)
        print("Plotted Embedding")
        
        w += 0.5
        
    fig.suptitle(f"TSNE Embedding of {config['dataset']} {config['name']}", fontsize=50)  
    fig.savefig(f"Plots/{config['dataset']}{config['suffix']}_{config['name']}.svg")
    plt.show()
    plt.close() 

def SpecialCovariances(dataset, labeldict, **kwargs):
    config.update(kwargs)
    
    fig, axes = plt.subplots(ncols=3, figsize=(45,15))
    
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    embedding['sizes'] = np.mean(config['size'])
    embedFlags(embedding, f'NORMAL embedding with w=1', ax=axes[0])
    print('Plotted subplot')
    
    Gaussians = Dataset2Gaussians(dataset, diagonal=True)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    embedding['sizes'] = np.mean(config['size'])
    embedFlags(embedding, f'embedding with DIAGONAL Covariance', ax=axes[1])
    print('Plotted subplot')
    
    Gaussians = Dataset2Gaussians(dataset, normalize=True)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    embedding['sizes'] = np.mean(config['size'])
    embedFlags(embedding, f'embedding with NORMALIZED Covariance', ax=axes[2])
    print('Plotted subplot')

    fig.suptitle(f"{config['dataset']} {config['name']} with Special Covariances", fontsize=50)  
    fig.savefig(f"Plots/{config['dataset']}{config['suffix']}_SpecialCovariance.svg")
    plt.show()
    plt.close() 

def Correlations(dataset, labeldict, rows, columns, normalize=True, selection=None, **kwargs):
    config.update(kwargs)
    
    fig, axes = plt.subplots(rows, columns, figsize=(10*columns,10*rows))
    
    Gaussians = Dataset2Gaussians(dataset, normalize=normalize)
    WSDM = GaussianWassersteinDistance(Gaussians)
    
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series().map(labeldict)
    embedding['sizes'] = np.mean(config['size'])

    axes = axes.flatten()
    
    if not selection:
        selection = it.combinations(dataset.columns, r=2)
        
    for ax, (feature1, feature2) in zip(axes.flatten(), selection):    
        corr = dataset.groupby(level=0).corr().fillna(0)
        sizes = corr.swaplevel().loc[feature1, feature2].values

        im=ax.scatter(embedding['x'], embedding['y'],
                   c=sizes, cmap='seismic', vmax=1, vmin=-1)
        ax.set_title(f'{feature1} with {feature2}', fontsize=25)
        ax.axis('off')
        print('Plotted Correlation')
         
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_ticks([-1,0,1])
    cbar.set_ticklabels(['anti', 'none', 'high'])
    cbar.ax.tick_params(labelsize=60)
    
    fig.suptitle(f"{config['dataset']} Feature Correlations", fontsize=100)  
    fig.savefig(f"Plots/{config['dataset']}{config['suffix']}_Correlations.svg")
    plt.show()
    plt.close()    

def Features(dataset, labeldict, FeatureSizes, **kwargs):
    config.update(kwargs)
    
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)

    for feature in FeatureSizes:    
        fig, ax = plt.subplots(figsize=(15,15))
    
        sizes = FeatureSizes[feature].values
        minsize, maxsize = config['size']
        minval, maxval   = sizes.min(), sizes.max()
        sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
        embedding['sizes'] = sizes

        embedFlags(embedding, f"{config['dataset']} Feature: {config['renaming'](feature)}", ax=ax)
        print("Plotted Feature")
        
        fig.savefig(f"Plots/{config['dataset']}{config['suffix']}_Features_{feature}.svg")
        plt.show()
        plt.close() 
        