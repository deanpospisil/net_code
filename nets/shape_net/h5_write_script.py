# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:15:17 2016

@author: dean
"""

import os
import numpy as np
import h5py
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'

nunits = 100
img_width = 32
n_imgs = 200
data = np.zeros((img_width, img_width, n_imgs))

targets = np.ones((n_imgs, nunits))


with h5py.File(top_dir + 'images/imagedb/solver_data.h5', 'w') as f:
    f['data'] = data
    f['targets'] = targets

with open(top_dir + 'images/imagedb/solver_data_list.txt', 'w') as f:
    f.write(top_dir + 'images/imagedb/solver_data.h5\n' )