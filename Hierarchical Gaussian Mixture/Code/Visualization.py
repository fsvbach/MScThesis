#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:12:56 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        # print(center, width, height, trans, fontsize)
        p = Ellipse(xy=center, width=width ,
                             height=height , angle=0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def plotHGM(mixture, prefix='TEST', std=1, X=20, Y=15, r=5):
    C = mixture.C
    fig, (ax, info) = plt.subplots(1,2, 
                                   figsize=(X,Y),
                                   gridspec_kw={'width_ratios': [r, 1],
                                                'wspace':0, 'hspace':0})
    
    # plotting grey samples
    xmeans, ymeans = mixture.data.reshape((-1,2)).T
    ax.scatter(xmeans, ymeans, s=1, c='grey', label='Samples')
    
    maxstd = 0
    datlabel = 'Datapoints (Distributions)'
    for i, data in enumerate(mixture.data):
        
        # plotting colourful datapoints
        xmeans, ymeans = data.mean(axis=1).T
        ax.scatter(xmeans, ymeans, s=25, c="C"+str(i), label=datlabel)
        datlabel=None
        
        # calculating largest data-covariance
        GaussianData = mixture.datapoints[i*mixture.N]
        _, width, height, _ = GaussianData.shape(std=1)
        maxstd = max(maxstd, max(width, height))
    
    # calculate geometry for info axis
    maxstd *= 7/6
    ratio = ax.get_data_ratio()
    ymax  = maxstd*ratio*r
    
    covlabel = f'{std}-σ class covariance'
    for i, data in enumerate(mixture.data):
    
        # plotting black class covariances
        GaussianClass = mixture.classes[i]
        mean, width, height, angle = GaussianClass.shape(std=std)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor='black', facecolor='none', 
                      linewidth=2, linestyle='--', 
                      label=covlabel)
        ax.add_patch(ell)
        covlabel=None
        
        # plotting legend data covariances
        GaussianData = mixture.datapoints[i*mixture.N]   
        _, width, height, angle = GaussianData.shape(std=1)
        ell = Ellipse(xy=(0,(i+0.5)*ymax/C), width=width, height=height, angle=angle, 
                      edgecolor="C"+str(i), facecolor='none', 
                      linewidth=3, 
                      label='class'+str(i+1))
        info.add_patch(ell)
        # plt.axis('off')
    
    info.yaxis.tick_right()
    info.set(xbound=(-maxstd/2,maxstd/2),
             ybound=(0,ymax))
    # info.set_aspect(2/C*ratio)
    
    ax.spines['right'].set(visible=False)
    info.spines['left'].set(linestyle=':', linewidth=0.5)
    
    axhandles, axlabels = ax.get_legend_handles_labels()
    infhandles, inflabels = info.get_legend_handles_labels()
    
    # axhandles[2].set_color('black')
    # infhandles[0].set_color('black')
    
    handles = [axhandles[1], axhandles[2], infhandles[0],axhandles[0]]
    labels = [axlabels[1], axlabels[2], '1-σ data covariance', axlabels[0]]
    ax.legend(handles, labels, 
                handler_map={Ellipse: HandlerEllipse()})
                # loc='lower left', bbox_to_anchor=(-0.5,1.05))
    
    fig.suptitle(mixture._info(), fontsize=24)

    fig.savefig(f"Plots/{prefix}_HGM.svg")
    plt.show()
    plt.close()
        
    
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class plotTSNE:
    def __init__(self, labels, prefix, k=10):
        self.labels = labels
        self.prefix = prefix
        self.params = []
        self.values = []
        self.embeddings = []
        self.classes = np.unique(labels)
        self.wplot  = [0, 0.5, 1]
        self.names  = ['Euclidean', 'Wasserstein', 'Covariance']
        self.kNN    = KNeighborsClassifier(k)
        self.k      = k
    
    def append(self, embedding, w):
        self.kNN.fit(embedding, self.labels)
        test = self.kNN.predict(embedding)
        acc  = accuracy_score(test, self.labels)
        self.values.append(acc)
        self.params.append(w)
        if w in self.wplot:
            self.embeddings.append(embedding)
        return acc
    
    def plot(self):
        fig, axes  = plt.subplots(ncols=4, figsize=(20,5))
        
        for ax, embedding, name, w in zip(axes[:-1], self.embeddings, self.names, self.wplot):
            ax.set(title=f"{name} TSNE embedding (w={w})",
                   aspect='equal')
    
            for c in self.classes:
                idx = np.where(self.labels==c)
                x, y = embedding[idx].T
                ax.scatter(x, y, s=1, c=f'C{c}')
        

        axes[-1].plot(self.params, 100*np.array(self.values))
        axes[-1].set(xlabel='w',
                     ylabel='%',
                     title =f"kNN Accuracies (k={self.k})",
                     ybound=(0,100),
                     xbound=(0,1))

        fig.savefig(f"Plots/{self.prefix}_TSNE.svg")
        plt.show()
        plt.close()
        
def plotMatrix(matrices, titles, name):
    n = len(matrices)
    fig, axes = plt.subplots(ncols=n, figsize=(7*n,5))
    for matrix, title, ax in zip(matrices, titles, axes):
        m = ax.imshow(matrix)
        ax.set(title=title)
        plt.colorbar(m, ax=ax)
    fig.savefig(f'Plots/Heatmaps/{name}.svg')
    plt.show()
    plt.close()