#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:04:54 2021

@author: fsvbach
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 12,
    'text.usetex': True,
    'pgf.rcfonts': False})

def embedScatter(embedding, title, size=1, ax=None):
    if not ax:
        ax = plt.gca()
        
    for label, data in embedding.groupby(level=0):
        X, Y = data['x'], data['y']
        ax.scatter(X, Y, s=size, label=label)

    ax.set_xticks([], minor=[])
    ax.set_yticks([], minor=[])
    # for spine in ['bottom', 'top', 'left', 'right']:
    #     ax.spines[spine].set_linestyle("dashed")
    ax.set_title(title)
    
class HandlerEllipseRotation(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       ydescent, xdescent, height, width, fontsize, trans):
        center = 0.5 * height - 0.5 * xdescent, 0.5 * width - 0.5 * ydescent
        p = Ellipse(xy=center, width=orig_handle.width,
                             height=orig_handle.height,
                             angle=orig_handle.angle)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
    
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
        
def plotMixture(mixture, std=1, ax=None):
    fig = None
    if not ax:
        fig, ax = plt.subplots(figsize=(10,10))
        
    # plotting grey samples
    xsample, ysample = mixture.data.values.T
    ax.scatter(xsample, ysample, s=1, c='grey', label='Samples')

    # flattening the dataset
    datlabel = 'Data'
    covlabel = rf'{std}-$\sigma$ class covariance'
    dataset  = mixture.data.groupby(level=0).mean()
    dataset.index = dataset.index.to_series().map(mixture.labeldict())
    
    for i, data in dataset.groupby(level=0):
        # plotting colourful datapoints
        xmeans, ymeans = data.values.T
        ax.scatter(xmeans, ymeans, s=15, c="C"+str(i), label=datlabel)
        datlabel=None

    for Gaussian in mixture.ClassGaussians:
        # plotting black class covariances
        mean, width, height, angle = Gaussian.shape(std=std)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor='black', facecolor='none', 
                      linewidth=2, linestyle='--', 
                      label=covlabel)
        ax.add_patch(ell)
        covlabel=None
    
    # storing 1st legend 
    leg1 = ax.legend(handler_map={Ellipse: HandlerEllipse()}, 
                     loc='lower left', title="Hierarchical Structure")
    
    handles = []
    labels  = []
    for i, Wishart in enumerate(mixture.ClassWisharts):
        # adding data covariances to 2nd legend 
        width, height, angle = Wishart.shape(std=3)
        ell = Ellipse(xy=(0,0), width=width, height=height, angle=angle, 
                      edgecolor="C"+str(i), facecolor='none', 
                      linewidth=1, 
                      label='class '+str(i+1))
        handles.append(ell)
        labels.append('Class ' +str(i))
        # ax.add_artist(ell)
        
    # adding legends
    ax.legend(handles, labels, handler_map={Ellipse: HandlerEllipseRotation()},
              title="Data Covariances")
    ax.add_artist(leg1)
    
    # add title
    ax.set_title(mixture.info)

    return fig

def plotWasserstein(Uhist, Vhist, D, opt_res):
    n,m = len(Uhist), len(Vhist)
    emd = round( opt_res.fun,3)
    gamma = opt_res.x.reshape((n, m))

    fig, axes = plt.subplots(2,3, figsize=(13,5), 
                             gridspec_kw={'width_ratios': (2*n/10,n,n),
                                          'height_ratios': (m/10,m)})
    [ax.set_axis_off() for ax in axes.ravel()]
    
    axes[0,1].bar(np.arange(n), Uhist, color='C0', alpha=0.5)
    axes[0,1].set(xlim=(-0.5,n-0.5))
    axes[1,0].barh(np.arange(m), Vhist, color='C1', alpha=0.5)
    axes[1,0].set(ylim=(-0.5,m-0.5))
    axes[1,0].invert_xaxis()
    axes[1,0].invert_yaxis()
    
    axes[1,1].imshow(gamma.T, cmap='Greys', vmin=0)
    axes[1,2].imshow(D.T, cmap='Greys', vmin=0)
    
    axes[0,2].text(0.5,0.5, f"scipy.linprog EMD={emd}", ha='center')
    fig.tight_layout()
    return fig

