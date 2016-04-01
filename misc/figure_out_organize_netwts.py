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


with open(top_dir + 'nets/netwts.p', 'rb') as f:
    try:
        a = pickle.load(f, encoding='latin1')
    except:
        a = pickle.load(f)
def frac_var_chrom(im):
    im = im - np.mean(im)
    print(im.shape)
    tot_var = np.linalg.norm(im)**2
    white_comp = np.tile(np.mean(im, 2, keepdims=True), (1,1,3))
    color_comp = im - white_comp

    color_var = np.linalg.norm(color_comp)**2
    return color_var/tot_var

# convert these to circular
layer = 0
ims = a[layer][1]
ims = np.array([im for im in ims])


sample_rate_mult = 1


#w, v =np.linalg.eig(cormat)
plt.figure()
data = ims
n = int(np.ceil(np.sqrt(data.shape[0])))
data = (data - data.min()) / (data.max() - data.min())

fclist = []
for ind in range(data.shape[0]):
    fclist.append(frac_var_chrom(np.swapaxes(data[ind],0,2)))
afc = np.argsort(fclist)

#for ind, kern in enumerate(afc):
for ind, kern in enumerate(range(len(afc))):

    plt.subplot(10, 10,ind+1)
    fc = frac_var_chrom(np.swapaxes(data[kern],0,2))
    _ = (data[kern] - data[kern].min()) / (data[kern].max() - data[kern].min())
    plt.imshow(np.swapaxes(_,0,2), interpolation='None')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(str(kern) +': '+ str(np.round(fc, decimals=2)))
plt.tight_layout()

plt.figure()
plt.hist(fclist)
plt.xlabel('Chromaticity')
plt.ylabel('Kernel Count')

