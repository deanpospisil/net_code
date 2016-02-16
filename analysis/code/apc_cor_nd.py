# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:54:30 2016

@author: dean
"""
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import dask.array as d
import seaborn as sns

top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)
sys.path.append( top_dir + '/xarray')

import xarray as xr
import d_misc as dm

dmod = xr.open_dataset(top_dir +'/data/apc_models.nc',chunks = {'models': 1000, 'shapes': 370}  )
da = xr.open_dataset( top_dir +'/data/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', chunks = {'unit': 100}  )



dmod = dmod.sel(models = range(1000), method = 'nearest' )
#unitsel = np.arange(0, da.dims['unit'], 1000)
#da = da.sel(unit = unitsel, method = 'nearest' )
##xsel = np.arange(-da.dims['x'] / 10., da.dims['x'] / 10., 2 )
da = da.sel(x = [0 , 1], method = 'nearest' )

#dm = dm.sel(models = range(100))

#using xray
#
def cor_apc_model(da, dmod, fn,fit_over_dims = None):
    #typically takes da, data, and dm, a set of linear models, an fn to write to,
    #and finally fit_over_dims which says over what dims is a models fit supposed to hold.

    da = da['resp'] - da['resp'].mean(('shapes'))
    dmod = dmod['resp']

    resp_n = da.vnorm(('shapes'))
    proj_resp_on_model = da.dot(dmod)

    if not fit_over_dims == None:
            resp_norm = resp_n.vnorm(fit_over_dims)
            proj_resp_on_model_var = proj_resp_on_model.sum(fit_over_dims)
            n_over = 0
            #count up how many unit vectors you'll be applying for each r.
            for dim in fit_over_dims:
                n_over = n_over + len(da.coords[dim].values)
    else:
        resp_norm =  resp_n
        proj_resp_on_model_var = proj_resp_on_model
        n_over = 1

    all_cor = (proj_resp_on_model_var) / (resp_norm*(n_over**0.5))

    cor = all_cor.max('models').load()

    sha = dm.provenance_commit(top_dir)
    cor.attrs['analysis'] = sha
    cor.attrs['model'] = dmod.attrs['model']
    cor.to_dataset(name = 'r').to_netcdf(fn)

    return cor

cor = cor_apc_model(da, dmod, fn = 'test_new_model_fit', fit_over_dims = ('x',))

#cwd + '/responses/apc_models_r_trans1.nc')






