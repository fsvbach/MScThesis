#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:45:29 2021

@author: fsvbach
"""

import matplotlib.pyplot as plt
import numpy as np


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

def plotEmbedding(ax, embedding, labels):
   
    for c in self.classes:
        idx = np.where(self.labels==c)
        x, y = embedding[idx].T
        ax.scatter(x, y, s=1, c=f'C{c}')