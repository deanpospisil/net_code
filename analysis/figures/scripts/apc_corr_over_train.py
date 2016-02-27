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
import apc_cor_nd as ac



dmod = xr.open_dataset(top_dir + 'analysis/data/models/apc_models.nc',
                       chunks = {'models': 1000, 'shapes': 370}  )
dmod = dmod.sel(models = range(10), method = 'nearest' )
dmod = dmod['resp']

ds = xr.open_mfdataset(top_dir + 'analysis/data/iter_*.nc',
                       concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
ds = ds.sel(unit = range(10),  method = 'nearest')


for iterind in ds.niter.values:
    da_c = da.sel(niter=iterind)
    cor = ac.cor_resp_to_model(da_c, dmod, fit_over_dims = ('x',))
    cor.to_dataset(name='r').to_netcdf(top_dir + 'analysis/data/r_iter_' + str(iterind) + '.nc')

ds = xr.open_mfdataset(top_dir + 'analysis/data/r_iter_*.nc', concat_dim = 'niter')
ds.to_netcdf(top_dir + 'analysis/data/r_iter_total_' + str(da.niter.shape[0]) +  '.nc')

