# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:28:06 2017

@author: deanpospisil
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import numpy as np

a = np.hstack((range(14), range(18, 318)));
a = np.hstack((a, range(322, 370)))
