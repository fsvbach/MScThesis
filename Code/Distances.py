#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:10:01 2021

@author: fsvbach
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# d = cdist(Y1, Y2)
# assignment = linear_sum_assignment(d)
# print(d[assignment].sum() / n)

def EuclideanDistanceMatrix(means):
    norms = np.linalg.norm(means, axis=1).reshape((len(means),1))**2
    return norms + norms.T - 2 * means@means.T