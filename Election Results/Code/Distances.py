#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:13:08 2021

@author: fsvbach
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

n = 5 

A = np.array([1,2,4,1,0,6]).reshape((3,2))

B = np.array([1,3,4,1,0,2]).reshape((3,2))

d = cdist(A, B)
assignment = linear_sum_assignment(d)
print(d[assignment].sum() / n)