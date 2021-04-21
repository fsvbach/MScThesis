#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:20:51 2021

@author: fsvbach
"""

import time

class Timer:
    def __init__(self, name, dec=3):
        self.name      = name
        self.dec       = 3
        self.start     = self.time()
        self.last_time = self.start
        self.date      = self.date()
        self.log       = [f"Started '{name}' at {self.date}\n"]
    
    def date(self):
        t = time.localtime()
        return time.strftime("%m-%d-%H-%M-%S", t)
    
    def time(self):
        return round(time.perf_counter(), self.dec)
    
    def total_time(self):
        return round(self.time() - self.start, self.dec)
    
    def add(self, infomsg):
        time = self.time()
        msg  = f'{infomsg} in {round(time-self.last_time, self.dec)}s. (Total: {self.total_time()}s)\n'
        self.last_time = time
        self.log.append(msg)
        print(msg)

    def finish(self):
        infomsg = f"Succesfully finished '{self.name}' in {self.total_time()}s"
        print(infomsg)
        for msg in reversed(self.log):
            infomsg = msg + '\n' + infomsg
        file = open(f"Plots/.{self.name} {self.date}.txt", "a")
        file.write(infomsg)
        file.close() 
        
        