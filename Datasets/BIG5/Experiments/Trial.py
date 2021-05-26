#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:42:26 2021

@author: fsvbach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



fig, ax  = plt.subplots()
ax.scatter()

ext = data.iloc[:,:10]
EXT = ext.melt(ignore_index=False).set_index('variable', append=True)




