# -*- coding: utf-8 -*-
"""
Created on Fri May 27 01:03:44 2016

@author: deanpospisil
"""

import sys
import os

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')

import xarray as xr
import numpy as np

da = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
