#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:07:28 2021

@author: bachmafy
"""

### run experiments by importing them and using their .run() method

from Code import EVS_TSNE as experiment
experiment.run()




############### Later ###################

# ### Figure 1 TSNE Plot HGM good
# from Code import HGM_TSNE as experiment
# experiment.run(seed=13, experiment="GOOD", datapoints=300)

# ### Figure 2 TSNE Plot HGM bad
# from Code import HGM_TSNE as experiment
# experiment.run(seed=13, experiment="BAD", datapoints=50)

# ### ...
