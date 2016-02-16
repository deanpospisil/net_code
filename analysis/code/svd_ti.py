# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:40:19 2016

@author: dean
"""

#svd analysis
import sys, os
import dask as dk
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)
sys.path.append( top_dir + '/xarray')
import xarray as xr
da = xr.open_dataset( top_dir + 'analysis/data/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', chunks = {'unit': 100}  )
d_da= da['resp'].data

d_da.shape
u, s, v = dk.array.linalg.svd(d_da[:,:,1])
