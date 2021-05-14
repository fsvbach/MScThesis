#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:13:59 2021

@author: bachmafy
"""

import matplotlib.pyplot as plt


def plotMatrix(matrices, titles, name):
    n = len(matrices)
    fig, axes = plt.subplots(ncols=n, figsize=(7*n,5))
    for matrix, title, ax in zip(matrices, titles, axes):
        m = ax.imshow(matrix)
        ax.set(title=title)
        plt.colorbar(m, ax=ax)
    fig.savefig(f'Plots/{name}.svg')
    plt.show()
    plt.close()


def plotHistogram(values, labels):
    alb = df.loc[df.country == i, question]
    alb = alb.loc[alb >= 0]
    alb.plot.hist()
    name = find(i)
    plt.title(name)
    plt.savefig(f'Plots/{name}.svg')
    plt.show()
    plt.close()

