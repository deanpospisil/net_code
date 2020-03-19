# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:36:19 2017

@author: deanpospisil
"""

import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
import pickle
layer_labels_b = [b'conv2', b'conv3', b'conv4', b'conv5', b'fc6']
layer_labels = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6']


top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + 'v4cnn'

with open(top_dir + '/data/an_results/ti_vs_wt_cov_exps_.p', 'rb') as f:    
    try:
        an = pickle.load(f, encoding='latin1')
    except:
        an = pickle.load(f)
        
with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
    try:
        netwtsd = pickle.load(f, encoding='latin1')
    except:
        netwtsd = pickle.load(f)
    