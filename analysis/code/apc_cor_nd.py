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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)
sys.path.append( cwd + '/xarray')
import xarray as xr


dm = xr.open_dataset(cwd +'/data/apc_models.nc',chunks = {'models': 1000, 'shapes': 370}  )
#da = xr.open_dataset( cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc',chunks = {'unit': 100, 'x': 100} )
#da = xr.open_dataset( cwd +'/responses/PC370_shapes_0.0_369.0_370.nc',chunks = {'unit': 100}  )
#da = xr.open_dataset( cwd +'/responses/PC370_shapes_matlab.nc',chunks = {'unit': 100}  )
da = xr.open_dataset( cwd +'/data/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', chunks = {'unit': 100}  )



#dm = dm.sel(models = range(100), method = 'nearest' )
#unitsel = np.arange(0, da.dims['unit'], 1000)
#da = da.sel(unit = unitsel, method = 'nearest' )
##xsel = np.arange(-da.dims['x'] / 10., da.dims['x'] / 10., 2 )
#da = da.sel(x = [ 0 ], method = 'nearest' )

#dm = dm.sel(models = range(100))

#using xray

def cor_apc_model(da, dm, fn):
    da = da['resp'] - da['resp'].mean(('shapes'))
    dm = dm['resp']
    
    resp_n =  da.vnorm(('shapes'))
    proj_resp_on_model = da.dot(dm)
    
    if 'x' in da.dims:
        resp_norm =  resp_n.vnorm(('x'))
        proj_resp_on_model_norm =  proj_resp_on_model.sum(('x'))
        n_x = len(da.coords['x'].values)
    else:
        resp_norm =  resp_n
        proj_resp_on_model_norm = proj_resp_on_model
        n_x = 1
        
    all_cor = (proj_resp_on_model_norm) / (resp_norm*(n_x**0.5))
    
    cor = all_cor.max('models')
    cor.to_dataset('r').to_netcdf(fn)
    
    return cor



#cwd + '/responses/apc_models_r_trans1.nc')






