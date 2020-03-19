# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:47:13 2016

@author: deanpospisil
"""

import sys
import numpy as np
import scipy.io as  l
import scipy
import scipy as sc
import matplotlib.pyplot as plt
import os
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append( top_dir + 'xarray/')

import d_curve as dc
import d_misc as dm
import d_img_process as imp
from scipy import ndimage


mat = l.loadmat(top_dir + 'net_code/img_gen/'+ 'PC3702001ShapeVerts.mat')
s = np.array(mat['shapes'][0])

with open(top_dir + 'net_code/data/models/PC370_params.p','rb') as f:
    curve = pickle.load(f)
