#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:07:28 2021

@author: bachmafy
"""

### run experiments by importing them and using their .run() method


from Datasets.BW2021.Data import Wahlkreise as Loader


data = Loader()

from WassersteinTSNE import HGM

data = HGM()



############### Later ###################

# ### Figure 1 TSNE Plot HGM good
# from Code import HGM_TSNE as experiment
# experiment.run(seed=13, experiment="GOOD", datapoints=300)

# ### Figure 2 TSNE Plot HGM bad
# from Code import HGM_TSNE as experiment
# experiment.run(seed=13, experiment="BAD", datapoints=50)

# ### Figure 2 CovarianceSampling 
# from Datasets.SyntheticData.Experiments import CovarianceSampling as experiment
# experiment.run(seed=2, suffix="TEST")

# ### Figure 3 Frobenius 
# from Datasets.SyntheticData.Experiments import Frobenius as experiment
# experiment.run(seed=2, suffix="FINAL")




