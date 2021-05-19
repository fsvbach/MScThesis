#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:04:54 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

class HandlerEllipseRotation(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent,
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
        
def plotHGM(ax, mixture, std=1):
    # plotting grey samples
    xsample, ysample = mixture.data.values.T
    ax.scatter(xsample, ysample, s=1, c='grey', label='Samples')

    # flattening the dataset
    datlabel = 'Data'
    covlabel = f'{std}-σ class covariance'
    dataset  = mixture.data.groupby(level=0).mean()
    dataset.index = mixture.labels
    
    for i, data in dataset.groupby(level=0):
        # plotting colourful datapoints
        xmeans, ymeans = data.values.T
        ax.scatter(xmeans, ymeans, s=25, c="C"+str(i), label=datlabel)
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
    leg1 = ax.legend(handler_map={Ellipse: HandlerEllipse()}, loc='lower left')
    
    handles = []
    labels  = []
    for i, Wishart in enumerate(mixture.ClassWisharts):
        # adding data covariances to 2nd legend 
        width, height, angle = Wishart.shape(std=3)
        ell = Ellipse(xy=(0,0), width=width, height=height, angle=angle, 
                      edgecolor="C"+str(i), facecolor='none', 
                      linewidth=3, 
                      label='class '+str(i+1))
        handles.append(ell)
        labels.append('Class ' +str(i))
        # ax.add_artist(ell)
        
    # adding legends
    ax.legend(handles, labels, handler_map={Ellipse: HandlerEllipseRotation()})
    ax.add_artist(leg1)

    return ax
        
