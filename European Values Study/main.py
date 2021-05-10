#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:15:58 2021

@author: fsvbach
"""


from openTSNE import TSNE
from Code.EVS import EVS
from Code.Visualization import plotEVS

import pandas as pd
import numpy as np

dataset = EVS()

countries = ['DE', 'SE', 'IT', 'HU', 'GB', 'RU', 'ES', 'BG',  'FR']

labels = []
NUTS2  = []

# for i, country in dataset.NUTS2(countries).groupby(level=0):
#     for j, nuts in country.groupby(level=1):
#         NUTS2.append(nuts.mean())
#         # NUTS?cov.append(nuts.cov())
#         labels.append(i)

# data = pd.concat(NUTS2, axis=1).T

# tsne = TSNE(random_state=13)

# embedding = tsne.fit(data.to_numpy())

# cord = pd.DataFrame(embedding, index=labels, columns=['x','y'])

embedding = pd.read_csv('Data/testcord.csv', index_col=0)

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import  Rectangle, Circle
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,20))

# plt.rcParams['legend.title_fontsize'] = 30

# for country, data in embedding.groupby(level=0):
#     ax.scatter(data['x'],data['y'], label=country)
#     ax.legend(title='NUTS2 regions from', fontsize=30, markerscale=4, 
#           ncol=2, bbox_to_anchor=(1.05,1.05))

# flag   = plt.imread(f'Data/flags/w640/{country.lower()}.png')
# im      = ax.imshow(flag, extent=[100,164,20,60])
# ax.set(xbound=[0,1000],
#         ybound=[0,300])
# plt.show()

path = f"Data/flags/w640/{'DE'.lower()}.png"

class AnyObject:
    pass


class AnyObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        img = plt.imread(path)
        imagebox = OffsetImage(img, zoom=.1)
        patch = AnnotationBbox(imagebox, [x0,y0],
                                xybox=(30, -30),
                                xycoords='data',
                                boxcoords="offset points")
        handlebox.add_artist(patch)
        return patch


plt.legend([AnyObject()], ['My first handler'],
            handler_map={AnyObject: AnyObjectHandler()})



# l=400
# w=300
# img = plt.imread(path)
# fig,ax = plt.subplots()
# rect = Rectangle((0, 0), l, w,facecolor='blue', edgecolor='k',
#                        linewidth=3,alpha=1, transform = ax.transData)
# im.set_clip_path(rect)
# ax.axis('off')
# plt.show()


# fig.savefig('Plots/tsneEVS.svg')
# plt.show()
# plt.close()