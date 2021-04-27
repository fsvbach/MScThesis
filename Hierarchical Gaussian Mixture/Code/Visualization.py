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

def plotHGM(mixture, prefix='TEST', std=1):
    C = mixture.C
    
    fig, (ax, info) = plt.subplots(1,2, 
                                   figsize=(15,10),
                                   gridspec_kw={'width_ratios': [3, 1],
                                                'wspace':0, 'hspace':0})
    
    # plotting grey samples
    xmeans, ymeans = mixture.data.reshape((-1,2)).T
    ax.scatter(xmeans, ymeans, s=1, c='grey', label='Samples')
    
    dist = 0
    datlabel = 'Datapoints (Distributions)'
    for i, data in enumerate(mixture.data):
        
        # plotting colourful datapoints
        xmeans, ymeans = data.mean(axis=1).T
        ax.scatter(xmeans, ymeans, s=25, c="C"+str(i), label=datlabel)
        datlabel=None
        
        # calculating largest data-covariance
        GaussianData = mixture.datapoints[i*mixture.N]
        _, width, height, _ = GaussianData.shape(std=1)
        dist = max(dist, max(width, height))
        
    covlabel = f'{std}-σ class covariances'
    for i, data in enumerate(mixture.data):
    
        # plotting black class covariances
        GaussianClass = mixture.classes[i]
        mean, width, height, angle = GaussianClass.shape(std=std)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor='black', facecolor='none', 
                      linewidth=1, linestyle='--', 
                      label=covlabel)
        ax.add_patch(ell)
        covlabel=None
        
        # plotting legend data covariances
        GaussianData = mixture.datapoints[i*mixture.N]   
        _, width, height, angle = GaussianData.shape(std=1)
        ell = Ellipse(xy=(0,(C-i-0.5)*dist), width=width, height=height, angle=angle, 
                      edgecolor="C"+str(i), facecolor='none', 
                      linewidth=3, 
                      label='class'+str(i+1))
        info.add_patch(ell)
        # plt.axis('off')
    
    
    fig.suptitle(mixture._info(), fontsize=24)
    
    info.set(xbound=(-dist,dist),
             ybound=(0,C*dist))
            # yticks=np.linspace(0,(n-1)*dist,num=n),
            # xticklabels=xlabel,
            # yticks=np.linspace(-2*dist/3,2*dist/3,num=5))
    info.yaxis.tick_right()
    info.legend(title='1-σ data covariances',
                handler_map={Ellipse: HandlerEllipse()})
    
    ax.legend(title='General HGM structure',
              handler_map={Ellipse: HandlerEllipse()})
            
    name = f"Plots/{prefix}_HGM"
    fig.savefig(f"{name}.svg")
    plt.show()
    plt.close()
    
    
def plotTSNE(TSNE, matrix, info, sklearn=False):
    C, N, w, seed, prefix = info
    
    metric='Wasserstein'
    if w == 0:
        metric='Euclidean'

    if sklearn:
        tsne = TSNE(metric='precomputed', 
                    square_distances=True, 
                    random_state=seed)
        embedding = tsne.fit_transform(matrix)
    else:
        tsne = TSNE(metric='precomputed', 
                    initialization='random', 
                    negative_gradient_method='bh',
                    random_state=seed)
        embedding = tsne.fit(matrix)

    for i in range(C):
        points = embedding[N*i:N*(i+1)]
        xmeans, ymeans = points.T
        plt.scatter(xmeans, ymeans, s=1)
    
    name = f"Plots/{prefix}_TSNE_{int(100*w)}"
    plt.title(f"{metric}TSNE embedding (w={w})")
    # plt.savefig(f"{name}.eps")
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close()
    
    return embedding
    
def plotMDS(MDS, matrix, info):
    C, N, w, prefix = info
    
    metric='Wasserstein'
    if w == 0:
        metric='Euclidean'

    mds = MDS(dissimilarity='precomputed')
    embedding = mds.fit_transform(matrix)

    for i in range(C):
        points = embedding[N*i:N*(i+1)]
        xmeans, ymeans = points.T
        plt.scatter(xmeans, ymeans, s=1)
    
    name = f"Plots/{prefix}_MDS_{int(100*w)}"
    plt.title(f"{metric}MDS embedding (w={w})")
    # plt.savefig(f"{name}.eps")
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close()
    
    
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class plotAccuracy:
    def __init__(self, labels, prefix, k=10):
        self.labels = labels
        self.prefix = prefix
        self.params = []
        self.values = []
        self.kNN    = KNeighborsClassifier(k)
    
    def append(self, embedding, w):
        self.kNN.fit(embedding, self.labels)
        test = self.kNN.predict(embedding)
        acc  = accuracy_score(test, self.labels)
        self.values.append(acc)
        self.params.append(w)
        return acc
    
    def plot(self):
        name = f"Plots/{self.prefix}_Accuracy"
        
        plt.plot(self.params, self.values)
        plt.xlabel('w')
        plt.ylabel('%')
        plt.title("kNN Accuracies")
        # plt.savefig(f"{name}.eps")
        plt.savefig(f"{name}.png")
        plt.show()
        plt.close()