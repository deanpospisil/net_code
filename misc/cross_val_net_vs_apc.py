# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 00:15:16 2016

@author: deanpospisil
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')


import d_img_process as imp
import xarray as xr

#cross validation comparison of APC and AlexNet
da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
daa=daa.loc[:, 0, :]#without translation
daa = daa.isel(shapes=da.coords['shapes'])