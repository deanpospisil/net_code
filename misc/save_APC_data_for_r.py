# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:33:08 2016

@author: deanpospisil
"""


import sys
import numpy as np
import scipy.io as  l
import scipy
import scipy as sc
import matplotlib.pyplot as plt
import os
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append( top_dir + 'xarray/')

import xarray as xr


with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
    apc_set = pickle.load(f)

v4 = xr.open_dataset(top_dir + 'net_code/data/responses/V4_362PC2001.nc')['resp']
apc_set = [ apc_set[int(ind)] for ind in v4.coords['shapes'].values ]

curve = np.hstack([shape['curvature'] for shape in apc_set])
ori = np.hstack([shape['orientation'] for shape in apc_set])
features = np.vstack([curve,ori])

resp = np.vstack([np.ones((len(shape['curvature']), 1))*v4.isel(shapes=i).values 
        for i, shape in enumerate(apc_set)])
    
np.savetxt(top_dir + 'net_code/data/models/shape_features.csv', features, delimiter=",")
np.savetxt(top_dir + 'net_code/data/responses/resp.csv', resp, delimiter=",")