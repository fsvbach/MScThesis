#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:46:36 2021

@author: fsvbach
"""

from Code.Visualization import plotHGM
from Code.Simulation import HierarchicalGaussianMixture

mixture = HierarchicalGaussianMixture(seed=11,
                                    datapoints=40, 
                                    samples=20, 
                                    features=2, 
                                    classes=4,
                                    ClassDistance=10,
                                    ClassVariance=5,
                                    DataVariance=1)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        # print(center, width, height, trans, fontsize)
        p = Ellipse(xy=center, width=width ,
                             height=height , angle=0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

