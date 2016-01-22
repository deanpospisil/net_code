# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:54:30 2016

@author: dean
"""
import numpy as np
import xray as xr
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import os, sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)


#best fit apc
# effect of blur
dm = xr.open_dataset(cwd +'/responses/apc_models.nc', chunks = {'models' : 7}  )
da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc', chunks={ 'unit': 7} )


unitsel = np.arange(0, da.dims['unit'], 1 )
da = da.sel(unit = unitsel, method = 'nearest' )

da_n = da - da.mean('shapes').mean('x')
da_n = da_n / ( ( da_n**2 ).sum('shapes').sum('x') )**0.5


dm_s = (da_n*dm).sum('shapes')*dm #get the projection of the apc vectors on each translation

#turn the vector of a scaled apc model over translations into a unit vector
dm_s = dm_s / ( ( dm_s**2 ).sum('shapes').sum('x') )**0.5 

#get the correlation of each of these scaled apc models
fitm = (da_n*dm_s).sum('shapes').sum('x')
fitm = fitm.chunk({'unit':100})
fitm = fitm.max('models')
print('chunked')

fitm.to_netcdf(cwd +'/responses/apc_models_r_trans.nc')
#fitm = xr.open_dataset(cwd +'/responses/apc_models_r_trans.nc' )
#
#b = fitm.to_dataframe()
#b.set_index(['layer_unit' ,'layer'], append=True, inplace=True)
#plt.close('all')
#
#sns.boxplot(x="layer_label", y="resp" , data=b[b['resp']>0], whis=1)
