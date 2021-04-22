#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 08:26:24 2021
@author: fsvbach
"""

import pandas as pd

df = pd.read_stata("Data/EVS_all.dta", convert_categoricals=False)

german_dim = df.loc[df.cntry == 276, 'E033']
german_dim = german_dim.loc[german_dim >= 0]

german = df.loc[df.cntry == 276, 'E012']
german.plot.hist()