#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:12:56 2021

@author: fsvbach
"""

eps = ""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
   
def plotHGMdata(mixture, prefix='TEST', std=1):
        
    classcolor = 'C1'
    datacolor = 'C0'
    samplecolor = 'C2'
    
    fig = plt.figure(figsize=(16,16))
    ax = plt.subplot(111, aspect='equal')

    xmeans, ymeans = mixture.data.reshape((-1,2)).T
    plt.scatter(xmeans, ymeans, s=1, c=samplecolor, label='samples')
    
    xmeans, ymeans = mixture.data_means().T
    plt.scatter(xmeans, ymeans, s=30, c=datacolor, label='datapoints')

    covlabel = f'{std}-σ class covariances'
    meanlabel = 'class means'
    for i in range(mixture.C):
        GaussianClass = mixture.classes[i]
                
        mean, width, height, angle = GaussianClass.shape(std=std)
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor=classcolor, facecolor='none', linewidth=3, 
                      label=covlabel, linestyle='--')
        ax.add_patch(ell)
               
        
        # GaussianData = mixture.datapoints[i*mixture.N]
        
        # mean, width, height, angle = GaussianData.shape(std=std)
        # ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
        #             edgecolor=datacolor, facecolor='none', linewidth=2 )
        # ax.add_artist(ell)
        
        xmean, ymean = GaussianClass.mean.T
        plt.scatter(xmean, ymean, s=100, c=classcolor, label=meanlabel)
        
        covlabel = meanlabel = None

    plt.legend()
    plt.title(mixture._info(), fontsize=25)
    
    name = f"Plots/{prefix}_HGMdata"
    # plt.savefig(f"{name}.eps")
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close()
    
def plotHGMclasses(mixture, prefix='TEST', std=1, n=3):
    
    fig = plt.figure(figsize=(16,16))
    ax = plt.subplot(111, aspect='equal')

    xmeans, ymeans = mixture.data.reshape((-1,2)).T
    plt.scatter(xmeans, ymeans, s=1, c='grey')
    
    for i, data in enumerate(mixture.data):
        color = "C"+str(i)
        
        # GaussianClass = mixture.classes[i]
        
        # mean, width, height, angle = GaussianClass.shape(std=std)
        # ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
        #               edgecolor='black', facecolor='none', linewidth=1, 
        #               linestyle='--')
        # ax.add_artist(ell)
        
        label=f'class {i+1}'
        for j in range(n):
            GaussianData = mixture.datapoints[i*mixture.N+j]
            
            mean, width, height, angle = GaussianData.shape(std=std)
            ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                        edgecolor=color, facecolor='none', linewidth=3, label=label)
            ax.add_patch(ell)
            
            xmeans, ymeans = data.mean(axis=1).T
            plt.scatter(xmeans, ymeans, s=30, c=color)
            label=None
    
        # xmean, ymean = GaussianClass.mean.T
        # plt.scatter(xmean, ymean, s=100, c='black')
        
    plt.legend(title=f'{std}-σ data covariances')
    plt.title(mixture._info(), fontsize=25)
    name = f"Plots/{prefix}_HGMlabels"
    # plt.savefig(f"{name}.eps")
    plt.savefig(f"{name}.png")
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