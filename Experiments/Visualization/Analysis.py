#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:39:49 2021

@author: fsvbach
"""


from WassersteinTSNE import Dataset2Gaussians, WassersteinTSNE, GaussianWassersteinDistance
from .utils import embedFlags, get_rectangle, border

import numpy as np
import matplotlib
import itertools as it
import matplotlib.pyplot as plt

_naming = {0: 'Euclidean', 0.5: 'Wasserstein', 1: 'Covariance'}

_config = {'folder': None,
           'dataset': None,
           'name': None,
           'w': 0.5,
           'size': (5,25),
           'seed': None,
           'description': '',
           'suffix': '',
           'renaming': lambda name: name}

def WassersteinEmbedding(dataset, labeldict, selection=None, **kwargs):
    config = {**_config, **kwargs}
    
    if not selection:
        selection = [0,0.5,1]
    M, N = get_rectangle(len(selection))
    fig, axes = plt.subplots(M,N, figsize=(25*N,25*M))
    
    for ax, w in zip(axes.flatten(), selection):
        Gaussians = Dataset2Gaussians(dataset)
        WSDM = GaussianWassersteinDistance(Gaussians)
        WT = WassersteinTSNE(WSDM, seed=config['seed'])
        embedding = WT.fit(w=w)
        embedding.index = embedding.index.to_series(name=config['folder']).map(labeldict)
        embedding['sizes'] = np.mean(config['size'])
        embedFlags(embedding, title=f"{_naming.get(w, '')} embedding (w={w})", ax=ax)
        if w == config['w']:
            border(ax, 'red')
        print("Plotted Embedding")
        
    # fig.suptitle(f"TSNE Embedding of {config['dataset']} {config['name']}", fontsize=48)  
    fig.tight_layout(pad=1)
    fig.savefig(f"Plots/{config['dataset']}{config['description']}_{config['name']}{config['suffix']}.svg")
    return fig

def SpecialCovariances(dataset, labeldict, **kwargs):
    config = {**_config, **kwargs}
    
    fig, axes = plt.subplots(ncols=3, figsize=(75,25))
        
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=1)
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    embedding['sizes'] = np.mean(config['size'])
    embedFlags(embedding, f'NORMAL embedding (w=1)', ax=axes[0])
    print('Plotted subplot')
    
    Gaussians = Dataset2Gaussians(dataset, diagonal=True)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=1)
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    embedding['sizes'] = np.mean(config['size'])
    embedFlags(embedding, f'DIAGONAL Covariance (Variance)', ax=axes[1])
    print('Plotted subplot')
    
    Gaussians = Dataset2Gaussians(dataset, normalize=True)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=1)
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    embedding['sizes'] = np.mean(config['size'])
    embedFlags(embedding, f'NORMALIZED Covariance (Correlation)', ax=axes[2])
    print('Plotted subplot')

    # fig.suptitle(f"{config['dataset']} {config['name']} with Special Covariances", fontsize=50)  
    fig.tight_layout(pad=1)
    fig.savefig(f"Plots/{config['dataset']}{config['description']}_SpecialCovariance{config['suffix']}.svg")
    return fig

def Correlations(dataset, labeldict, normalize=True, selection=None, **kwargs):
    config = {**_config, **kwargs}
    
    Gaussians = Dataset2Gaussians(dataset, normalize=normalize)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series().map(labeldict)
    embedding['sizes'] = np.mean(config['size'])

    if not selection:
        selection = it.combinations(dataset.columns, r=2)
    selection = list(selection)
    M, N = get_rectangle(len(selection))
    fig, axes = plt.subplots(M,N, figsize=(10*N,10*M))
    # fig.tight_layout(pad=0.51)
    
    for ax, (feature1, feature2) in zip(axes.flatten(), selection):    
        corr = dataset.groupby(level=0).corr().fillna(0)
        sizes = corr.swaplevel().loc[feature1, feature2].values

        im=ax.scatter(embedding['x'], embedding['y'],
                   c=sizes, cmap='seismic', vmax=1, vmin=-1)
        ax.set_title(f'{feature1} with {feature2}', fontsize=45)
        ax.axis('off')
        print('Plotted Correlation')
         
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_ticks([-1,0,1])
    cbar.set_ticklabels(['anti', 'none', 'high'])
    cbar.ax.tick_params(labelsize=60)
    
    fig.suptitle(f"{config['dataset']} Feature Correlations {config['suffix']} with embedding w={config['w']}", fontsize=100)  
    fig.savefig(f"Plots/{config['dataset']}{config['description']}_Correlations{config['suffix']}.svg")
    return fig  

def Features(dataset, labeldict, FeatureSizes, selection=False, **kwargs):
    config = {**_config, **kwargs}
    
    Gaussians = Dataset2Gaussians(dataset)
    WSDM = GaussianWassersteinDistance(Gaussians)
    WT = WassersteinTSNE(WSDM, seed=config['seed'])
    embedding = WT.fit(w=config['w'])
    embedding.index    = embedding.index.to_series(name=config['folder']).map(labeldict)
    
    if selection:
        N = len(FeatureSizes.columns)
        fig, axes = plt.subplots(1,N, figsize=(20*N,20))
    
        for ax, feature in zip(axes.flatten(), FeatureSizes):
            sizes = FeatureSizes[feature].values
            minsize, maxsize = config['size']
            minval, maxval   = sizes.min(), sizes.max()
            sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
            embedding['sizes'] = sizes
    
            embedFlags(embedding, f"{config['renaming'](feature)}", ax=ax)
            print("Plotted Feature")
            
        fig.suptitle(f"{config['dataset']} Feature {config['suffix']} with embedding w={config['w']}", fontsize=50) 
        fig.savefig(f"Plots/{config['dataset']}{config['description']}_Features{config['suffix']}.svg")
        return fig
    
    for feature in FeatureSizes:    
        fig, ax = plt.subplots(figsize=(20,20))
    
        sizes = FeatureSizes[feature].values
        minsize, maxsize = config['size']
        minval, maxval   = sizes.min(), sizes.max()
        sizes = minsize + (sizes-minval)*(maxsize-minsize)/(maxval-minval)
        embedding['sizes'] = sizes

        embedFlags(embedding, f"{config['dataset']} Feature {config['suffix']}: {config['renaming'](feature)}", ax=ax)
        print("Plotted Feature")
        
        fig.savefig(f"Plots/{config['dataset']}{config['description']}_Features_{feature}{config['suffix']}.svg")
        plt.show()
        plt.close() 