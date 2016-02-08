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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)
sys.path.append( cwd + '/xarray')
import xarray as xr



dm = xr.open_dataset(cwd +'/responses/apc_models.nc',chunks = {'models': 1000, 'shapes': 370}  )
da = xr.open_dataset( cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc',chunks = {'unit': 100, 'x': 100} )

#dm = dm.sel(models = range(100), method = 'nearest' )
#
#unitsel = np.arange(0, da.dims['unit'], 1000)
#da = da.sel(unit = unitsel, method = 'nearest' )
#
#
##xsel = np.arange(-da.dims['x'] / 10., da.dims['x'] / 10., 2 )
#da = da.sel(x = [ 0,], method = 'nearest' )
#
##dm = dm.sel(models = range(100))
#
#t = da['resp'].dot( dm['resp'] )
##t is the projection of each apc model onto each of the translated responses
#
#n = da['resp'].vnorm(['shapes', 'x'] )
#t = t.vnorm( 'x' )
#
#cor = (t/n).max('models')
#
#cor.to_dataset('cor').to_netcdf(cwd +'/responses/apc_models_r.nc')

fitm = xr.open_dataset(cwd +'/responses/apc_models_r_trans.nc' )
#fitm = xr.open_dataset(cwd +'/responses/apc_models_r.nc' )
###

b = fitm.to_dataframe()
b.set_index(['layer_unit' ,'layer'], append=True, inplace=True)
plt.close('all')



#sns.boxplot(x="layer_label", y="cor" , data=b)
per = b[b>0.5].groupby('layer_label').count()/b.groupby('layer_label').count()
per.plot(kind = 'bar')
plt.ylim((0,.3))
plt.title('Percent units > 0.5 Correlation across Translation')
fitm = xr.open_dataset(cwd +'/responses/apc_models_r.nc' )
#fitm = xr.open_dataset(cwd +'/responses/apc_models_r.nc' )
###

b = fitm.to_dataframe()
b.set_index(['layer_unit' ,'layer'], append=True, inplace=True)


#sns.boxplot(x="layer_label", y="cor" , data=b)
per = b[b>0.5].groupby('layer_label').count()/b.groupby('layer_label').count()
per.plot(kind = 'bar')
plt.ylim((0,.3))
plt.title('Percent units > 0.5 Correlation at Single Position')
