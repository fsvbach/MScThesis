#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:50:43 2021

@author: bachmafy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from WassersteinTSNE.Distances import GaussianWassersteinDistance, linprogSolver
from WassersteinTSNE.Distributions import GaussianDistribution
from Experiments.Visualization import Analysis, utils
from Datasets.GER2017 import Bundestagswahl

GER = Bundestagswahl(numparty=6)
names = GER.labeldict('Gebiet')
dataset = GER.data

wahlkreise = ['Berlin-Neukölln', 'Hamburg-Mitte']
features = ['DIE LINKE', 'AfD']

# wahlkreise = ['Kiel', 'Dresden I']
# features = ['SPD', 'AfD']

dataset.index = dataset.index.to_series().map(names)

U = dataset.loc[wahlkreise[0], features]
V = dataset.loc[wahlkreise[1], features]

G = GaussianDistribution()
Gaussians = [G.estimate(U), G.estimate(V)]

WSDM = GaussianWassersteinDistance(pd.Series(Gaussians, index=wahlkreise))
diff_gauss = WSDM.matrix()[1,0]
diff_means = WSDM.matrix(w=0)[1,0]
diff_covs = WSDM.matrix(w=1)[1,0]    
                    
opt_res = linprogSolver(U, V)
diff_exact = np.sqrt(-opt_res.fun )

fig, ax = plt.subplots(figsize=(7,7))

for G, data, name, color in zip(Gaussians, [U,V], wahlkreise, ['C0','C1']):

    correlation = round(data.corr().iloc[1,0],2)
    X, Y = data.values.T
    ax.scatter(X, Y, s=10, c=color)

    utils.plotGaussian(G, size=0, color=color, STDS=[2], ax=ax)
    
    x,y = G.mean
    ax.scatter(x,y,label=f'{name} \n Corr: {correlation}',c=color,s=200)
  
ax.set_aspect('equal')    
ax.set(title='Correlation of poll stations within a voting district',
       xlabel=features[0],
       ylabel=features[1])
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# place a text box in upper left in axes coords
textstr = '\n'.join((
    r'$d_{{Covs}}=%.4f$' % (diff_covs, ),
    r'$d_{{Means}}=%.4f$' % (diff_means, ),
    r'$d_{{Gauss}}=%.4f$' % (diff_gauss, ),
    r'$d_{{Exact}}=%.4f$' % (diff_exact, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.75, textstr, transform=ax.transAxes, fontsize=14,
        va='top', ha='right', bbox=props)

ax.legend(loc='best')

fig.tight_layout()
fig.savefig(f"Plots/GER_Correlation_{''.join(wahlkreise)}.svg")
fig.savefig(f"Reports/Figures/GER/JointCorrelation_LinksAfD.pdf")
plt.show()
plt.close()