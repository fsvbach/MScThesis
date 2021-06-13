#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:34:04 2021

@author: fsvbach
"""

from Experiments.Visualization import Analysis 
from Experiments.utils import MeanStdCorr
from Datasets.GER2017 import Wahlbezirke

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

GER     = Wahlbezirke(numparty=6)
MeanStdCorr(GER.data, ax=ax1, title='Without transformation')
MeanStdCorr(GER.transform(), ax=ax2, title='After transformation')

fig.tight_layout()
fig.savefig('Plots/GER_VarStab.svg')
fig.savefig('Reports/Figures/GER/MeanStdCorr.pdf')
plt.show()
plt.close()


labels  = GER.labels.Bundesland.to_dict()
Analysis._config.update(folder='wappen', 
                        seed=13, 
                        name='Wahlkreise',
                        description='max6',
                        size=(10,30),
                        dataset='GER',
                        w=0.75)

fig = Analysis.WassersteinEmbedding(GER.data, labels, 
                              selection=[0,0.5,0.75,0.875,0.9475,1], 
                              suffix='trafo')
fig.savefig('Reports/Figures/GER/Transformation.pdf')
plt.show()