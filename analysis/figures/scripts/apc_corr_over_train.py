# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:39:42 2016

@author: dean
"""
import os, sys
import numpy as np

top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)
sys.path.append( top_dir + '/xarray')
import xarray as xr
import d_misc as dm
import pandas as pd
import matplotlib.pyplot as plt

dmod = xr.open_dataset(top_dir + 'analysis/data/r_iter_total_10.nc' ).load()
dmod = xr.open_dataset(top_dir + 'analysis/data/apc_models_r_trans.nc' ).load()
dmod = dmod.reindex( { 'niter': np.sort(dmod.coords['niter'].values )})


r=[]
for niter in dmod.coords['niter'].values:
    r.append(np.sum(np.squeeze(dmod.sel(niter=niter).to_array().values>0.5)))

plt.stem(r)

np.sum(dmod['cor'].values>0.5)