#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:53:24 2021

@author: bachmafy
"""

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
   

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        print(center, width, height, trans, fontsize)
        p = mpatches.Ellipse(xy=center, width=width ,
                             height=height , angle=45)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


c = mpatches.Ellipse(xy=(0.5,0.5), width=0.2,
                             height=0.11,angle=45)
# plt.gca().add_patch(c)

# plt.legend([c], ["An ellipse, not a rectangle"])

plt.legend([c], ["An ellipse, not a rectangle"],
            handler_map={mpatches.Ellipse: HandlerEllipse()})