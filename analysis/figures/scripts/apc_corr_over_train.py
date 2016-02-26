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



da = xr.open_dataset( top_dir + 'analysis/data/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', chunks = {'unit': 100}  )

dmod = dmod.sel(models = range(10), method = 'nearest' )
ds = xr.open_mfdataset(top_dir + 'analysis/data/iter_*.nc',
                       concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
da = ds.to_array().chunk(chunks = {'niter':1, 'unit':100, 'shapes': 370})
da = da.sel(x = np.linspace(-50, 50, 2), method = 'nearest' )
da = da.sel(niter = np.linspace(0, da.coords['niter'].shape[0], 2),
                                method = 'nearest')
da = da.sel(unit = range(10),  method = 'nearest')


for iterind in ds.niter.values:
    da_c = da.sel(niter=iterind)
    cor = cor_resp_to_model(da_c, dmod, fit_over_dims = ('x',))
    cor.to_dataset(name='r').to_netcdf(top_dir + 'analysis/data/r_iter_' + str(iterind) + '.nc')

ds = xr.open_mfdataset(top_dir + 'analysis/data/r_iter_*.nc', concat_dim = 'niter')
ds.to_netcdf(top_dir + 'analysis/data/r_iter_total_' + str(da.niter.shape[0]) +  '.nc')

