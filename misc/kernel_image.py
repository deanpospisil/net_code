# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:58:47 2016

@author: deanpospisil
"""
import pickle
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt



top_dir = os.getcwd().split('net_code')[0]
top_dir = top_dir+ 'net_code/'

if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:
        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
layer = 0    
ims = a[layer][1]
ims = np.array([im for im in ims])
low_rank = []
cind = 7
im = ims[cind,:,:]
im = np.swapaxes(im, 0, 2)
im = im.astype(np.float64)
im=im/np.linalg.norm(im)
imn = im - np.min(im)
imn = imn / np.max(imn)

plt.imshow(imn, interpolation='nearest')
plt.xticks([])
plt.yticks([])