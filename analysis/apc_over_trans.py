# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:54:30 2016

@author: dean
"""
import numpy as np
import xray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import dask.array as d


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)


#best fit apc
# effect of blur
dm = xr.open_dataset(cwd +'/responses/apc_models.nc'  )
da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc' )


unitsel = np.arange(0, da.dims['unit'], 10 )
da = da.sel(unit = unitsel, method = 'nearest' )

xsel = np.arange(-da.dims['x'] / 10., da.dims['x'] / 10., 2 )
da = da.sel(x = [8, 6, 4, 2,0,2,4,6 ,8], method = 'nearest' )

x = np.transpose(d.from_array(da['resp'].values, chunks=(1000, 1,1000)), (2,1,0))
y = d.from_array(dm['resp'].values, chunks=(370, 200 ))

t = x.dot(y)

#d.to_hdf5(cwd +'/responses/apc_models_r_trans.nc', {'/t': t})
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
