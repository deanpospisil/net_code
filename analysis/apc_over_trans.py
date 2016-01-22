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

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)


#best fit apc
# effect of blur
dm = xr.open_dataset(cwd +'/responses/apc_models.nc', chunks={'models': 100} )
da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc', chunks={'x': 1, 'unit':100} )

unitsel = np.arange(0, da.dims['unit'], 10 )
da = da.sel(unit = unitsel, method = 'nearest' )

unitsel = np.arange(0, da.dims['unit'], 100 )
da = da.sel(x = [4, 2, 0 ,2 ,4 ], method = 'nearest' )

da = da.chunk(chunks={'x': 1, 'unit':100})


da_n = da - da.mean('shapes').mean('x')
da_n = da_n / ( ( da_n**2 ).sum('shapes').sum('x') )**0.5

dm = dm / da_n.dims['x']**0.5 #need to adjust variance, performing correlation over translations


fitm = (da_n*dm).sum('shapes').sum('x').max('models')


fitm.to_netcdf(cwd +'/responses/apc_models_r_trans.nc')
fitm = xr.open_dataset(cwd +'/responses/apc_models_r_trans.nc' )

b = fitm.to_dataframe()
b.set_index(['layer_unit' ,'layer'], append=True, inplace=True)
plt.close('all')

sns.boxplot(x="layer_label", y="resp" , data=b[b['resp']>0], whis=1)