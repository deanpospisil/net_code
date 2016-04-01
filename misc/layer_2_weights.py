
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:12:06 2016

@author: deanpospisil
"""

import pickle
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import d_img_process as di
import d_misc as dm

if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
layer = 1  

ims = a[layer][1][:128]
##ims = ims / np.sqrt(np.sum(ims**2, 1, keepdims=True))
#kernsum = np.sum(ims, (2,3))
#plt.figure()
#n, bins, patches = plt.hist(kernsum.ravel(), bins=int(len(kernsum.ravel())/20),
#                            normed=False)
xims = xr.DataArray(ims, dims=['output', 'input', 'r', 'c'])
xims = xims.sum(('r','c'))
xims=xims.chunk().data
