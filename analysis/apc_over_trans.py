# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:54:30 2016

@author: dean
"""
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import dask.array as d

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)
sys.path.append( cwd + '/xarray')
import xarray as xr



dm = xr.open_dataset(cwd +'/responses/apc_models.nc',chunks = {'models': 1000, 'shapes': 370}  )
da = xr.open_dataset( cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc',chunks = {'unit': 100, 'x': 100} )
#dm = xr.open_dataset(cwd +'/responses/apc_models.nc'  )
#da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc')
#dm = dm.load()
#da = da.load()

dm = dm.sel(models = range(100), method = 'nearest' )

unitsel = np.arange(0, da.dims['unit'], 1000)
da = da.sel(unit = unitsel, method = 'nearest' )


xsel = np.arange(-da.dims['x'] / 10., da.dims['x'] / 10., 2 )
da = da.sel(x = [8, 6, 4, 2, 0, 2, 4, 6, 8], method = 'nearest' )

dm = dm.sel(models = range(100))

t = da['resp'].dot( dm['resp'] )
#t is the projection of each apc model onto each of the translated responses

n = da['resp'].vnorm(['shapes', 'x'] )
t = t.vnorm( 'x' )

cor = (t/n).max('models')



cor.to_hdf5(cwd +'/responses/apc_models_r_trans.nc', {'/t': t})
#da_n = da - da.mean('shapes').mean('x')
#da_n = da_n / ( ( da_n**2 ).sum('shapes').sum('x') )**0.5
#
#
#dm_s = (da_n*dm).sum('shapes')*dm #get the projection of the apc vectors on each translation
#
##turn the vector of a scaled apc model over translations into a unit vector
#dm_s = dm_s / ( ( dm_s**2 ).sum('shapes').sum('x') )**0.5 
#
##get the correlation of each of these scaled apc models
#fitm = (da_n*dm_s).sum('shapes').sum('x').max('models')
#
#print('chunked')
#
#fitm.to_netcdf(cwd +'/responses/apc_models_r_trans.nc')
#fitm = xr.open_dataset(cwd +'/responses/apc_models_r_trans.nc' )
####
#
#b = fitm.to_dataframe()
#b.set_index(['layer_unit' ,'layer'], append=True, inplace=True)
#plt.close('all')
#
#
#sns.boxplot(x="layer_label", y="resp" , data=b)
dm.std