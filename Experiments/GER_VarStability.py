#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:34:04 2021

@author: fsvbach
"""

import numpy as np
from Experiments.Visualization import Analysis, utils
from Datasets.GER2017 import Bundestagswahl

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,2, figsize=(10,10))

GER     = Bundestagswahl(numparty=6)


utils.MeanStdCorr(GER.data.multiply((GER.size), axis='rows'), ax=axes[0,0], title=r'Absolute votes ($Y$)')
utils.MeanStdCorr(GER.data, ax=axes[0,1], title=r'Percentages ($\frac{Y}{n}$)')
utils.MeanStdCorr(GER.transform().divide(np.sqrt(GER.size), axis='rows'), ax=axes[1,0], title=r'$T(Y) = \arcsin(\sqrt{\frac{Y}{n}})$')
utils.MeanStdCorr(GER.transform(), ax=axes[1,1], title=r'$T(Y) = \sqrt{n} \arcsin(\sqrt{\frac{Y}{n}})$')

fig.tight_layout()
fig.savefig('Plots/GER_VarStab.svg')
fig.savefig('Reports/Figures/GER/MeanStdCorr.pdf')
plt.show()
plt.close()


# labels  = GER.labels()
# Analysis._config.update(folder='wappen', 
#                         seed=13, 
#                         name='Wahlkreise',
#                         description='max6',
#                         size=(10,30),
#                         dataset='GER',
#                         w=0.75)

# fig = Analysis.WassersteinEmbedding(GER.data, labels, 
#                               selection=[0,0.5,0.75,0.875,0.9475,1], 
#                               suffix='trafo')
# fig.savefig('Reports/Figures/GER/Transformation.pdf')
# plt.show()

