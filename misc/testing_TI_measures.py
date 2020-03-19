# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:36:44 2016

@author: deanpospisil
"""

#comparing methods

import numpy as  np
import scipy.io as  l
import os, sys
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
import pandas as pd

fn = top_dir +'data/responses/V4_362PC2001.nc'
v4 = xr.open_dataset(fn)['resp'].load()
