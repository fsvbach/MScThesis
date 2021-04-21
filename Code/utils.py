#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:20:51 2021

@author: fsvbach
"""

import time

class logprint:
    def __init__(self, name):
        self.name  = name
        self.start = time.perf_counter()
        self.log   = []
        
    def print(self, infomsg):
        time = time.perf_counter()

        