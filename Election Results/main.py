#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:29:45 2021

@author: bachmafy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Data/BW/Kreise.csv', delimiter=';', encoding='ISO-8859-1')

parties = data.columns[9:]

parties

