# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:12:56 2016

@author: dean
"""

import scipy.io as l
import os, sys
import numpy as np
import pickle

top_dir = os.getcwd().split('net_code')[0] 
sys.path.append(top_dir + 'net_code/common/')
sys.path.append( top_dir + 'xarray/')

import xarray as xr
import apc_model_fit as ac

#put responses into xarray
m = l.loadmat(top_dir + '/net_code/data/responses/pyresponses.mat')
m = m['pyresponses']

shape_id = m[:, 0]
resp = m[:, 1:]
da = xr.DataArray(resp, dims=['shapes', 'unit'], coords= [shape_id, range(resp.shape[1])])

ds = da.to_dataset(name='resp')
ds.to_netcdf(top_dir + 'net_code/data/responses/alex_bowl.nc' )

#open those responses, and build apc models for their shapes
with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

da = xr.open_dataset(top_dir + 'net_code/data/responses/alex_bowl.nc', chunks = {'shapes':370})['resp']

shape_id = da.coords['shapes'].values

shape_dict_list = [shape_dict_list[sn] for sn in shape_id.astype(int)]

maxAngSD = np.deg2rad(171)
minAngSD = np.deg2rad(23)
maxCurSD = 0.98
minCurSD = 0.09
nMeans = 16
nSD = 16
fn = 'apc_models_bowl.nc'
dam = ac.make_apc_models(shape_dict_list, shape_id, fn, nMeans, nSD, maxAngSD, minAngSD,
                      maxCurSD, minCurSD, prov_commit=False)

#load the models you made, and fit them to the cells responses
dmod = xr.open_dataset(top_dir + 'net_code/data/models/apc_models_bowl.nc',
                       chunks = {'models': 1000, 'shapes': 370}  )['resp']
da = xr.open_dataset(top_dir + 'net_code/data/responses/alex_bowl.nc', 
                     chunks = {'shapes': 370})['resp']
cor = ac.cor_resp_to_model(da, dmod, fit_over_dims=None, prov_commit=False)
