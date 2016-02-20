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
img_width = 32s
n_imgs = 200
data = np.zeros((n_imgs, 1, img_width, img_width))
data = data.astype('float32')

targets = np.ones((n_imgs, nunits))
targets = targets.astype('float32')

with h5py.File(top_dir + 'images/imagedb/train_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = targets

with open(top_dir + 'nets/shape_net/train_data_list.txt', 'w') as f:
    f.write(top_dir + 'images/imagedb/train_data.h5\n' )
    
    
with h5py.File(top_dir + 'images/imagedb/test_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = targets

with open(top_dir + 'nets/shape_net/test_data_list.txt', 'w') as f:
    f.write(top_dir + 'images/imagedb/test_data.h5\n' )