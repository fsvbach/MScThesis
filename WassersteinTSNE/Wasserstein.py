#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:57:05 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np

from openTSNE import TSNE as openTSNE

def EuclideanDistance(A,B):
    N1 = np.linalg.norm(A, ord=2, axis=1).reshape(-1,1)**2 
    N2 = np.linalg.norm(B, ord=2, axis=1).reshape(1,-1)**2 
    N3 = -2 * np.inner(A,B)
    D  = N1 + N2 + N3
    D[np.where(D<0)] = 0
    return np.sqrt(D)
    
def ConstraintMatrix(n,m):
    N = np.repeat(np.identity(n), m, axis=1)
    M = np.hstack([np.identity(m)]*n)
    return np.vstack([N,M])

