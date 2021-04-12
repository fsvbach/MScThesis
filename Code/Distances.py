#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:10:01 2021

@author: fsvbach
"""

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

d = cdist(Y1, Y2)
assignment = linear_sum_assignment(d)
print(d[assignment].sum() / n)
