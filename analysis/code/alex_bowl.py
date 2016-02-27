# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:12:56 2016

@author: dean
"""

import scipy.io as l
import os
import pickle
import numpy as np
import sys
top_dir = os.getcwd().split('net_code')[0] + '/net_code/'
sys.path.append(top_dir)
sys.path.append( top_dir + '/xarray')
import xarray as xr


m = l.loadmat(top_dir + 'analysis/data/pyresponses.mat')
m = m['pyresponses']

shape_id = m[:, 0]
resp = m[:, 1:]

da = xr.DataArray(resp, dims=['shapes', 'unit'], coords= [shape_id, range(resp.shape[1])])

ds = da.to_dataset(name = 'resp')
ds.to_netcdf(top_dir +'analysis/data/alex_bowl.nc' )

import apc_cor_nd as ac

dmod = xr.open_dataset(top_dir + 'analysis/data/models/apc_models_bowl.nc',
                       chunks = {'models': 1000, 'shapes': 370}  )['resp']
da = xr.open_dataset(top_dir + 'analysis/data/alex_bowl.nc', chunks = {'shapes': 370})['resp']
cor = ac.cor_resp_to_model(da, dmod)

bf = dmod.sel(models=cor[0].coords['models']).load()
r = da[:, 0].load()

import matplotlib.pyplot as plt
plt.plot(bf)
plt.plot((r-np.mean(r))/30.)


np.corrcoef([bf, r-np.mean(r)])