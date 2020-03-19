# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 21:20:26 2016

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


alex = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc')['resp']
