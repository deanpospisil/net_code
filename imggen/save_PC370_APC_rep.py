# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:03:26 2016

@author: deanpospisil
"""
import os
import pickle
import numpy as np
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
import scipy.io as l

mat = l.loadmat(top_dir + 'analysis/data/models/PC2001370Params.mat')
s = mat['orcurv'][0]

#adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]

a = np.hstack((range(14), range(18,318)))
a = np.hstack((a, range(322, 370)))
s = s[a]
'''

shape_dict_list = [{'curvature':None, 'orientation':None} ]

shape_dict_list = [{'curvature':params[:,1], 'orientation':params[:,0]} for params in s]
with open(top_dir + 'analysis/data/models/PC370_params_un.p','wb') as f:
    pickle.dump(shape_dict_list, f )

with open(top_dir + 'analysis/data/models/PC370_params_un.p','rb') as f:
    sh = pickle.load( f )


