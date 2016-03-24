# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:41:52 2016

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)

def frac_var_chrom(im):
    tot_var = np.linalg.norm(im)**2
    white_comp = np.tile(np.mean(im, 2, keepdims=True), (1,1,3))
    color_comp = im - white_comp
    color_var = np.linalg.norm(color_comp)**2    
    return color_var/tot_var

# convert these to circular
layer = 0    
ims = a[layer][1]
ims = np.array([im for im in ims])
im = ims[95,:,:]
im = np.swapaxes(im, 0,2)
im = im - np.mean(im)
im=im.astype(np.float64)
#plt.close('all')
plt.imshow((im - im.min()) / (im.max() - im.min()), interpolation='nearest')

tot_mag = np.linalg.norm(im)**2
white_comp = np.tile(np.mean(im, 2, keepdims=True), (1,1,3))
color_comp = im - white_comp

print(np.isclose(im, (white_comp+color_comp)).all())
#color_comp = color_comp.T
#plt.imshow((color_comp - color_comp.min()) / (color_comp.max() - color_comp.min()), interpolation='nearest')

print(np.linalg.norm(white_comp)**2)
print(np.linalg.norm(color_comp)**2)
print(np.linalg.norm(white_comp)**2 + np.linalg.norm(color_comp)**2)
print(tot_mag)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rgb = np.reshape(color_comp, (11*11, 3))
ax.scatter(rgb[:, 0], 
           rgb[:, 1], 
           rgb[:, 2])
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
rg=np.max(np.abs(im))
ax.set_xlim3d(-rg, rg)
ax.set_ylim3d(-rg, rg)

rgb = np.reshape(color_comp, (11*11, 3))
rgb = rgb - np.mean(rgb, 1, keepdims=True)
cov = np.dot(rgb.T, rgb)
w, v = np.linalg.eig(cov)

ax.plot(xs=[v[0, 0], -v[0, 0]], ys=[v[1, 0], -v[1, 0]], zs=[v[2, 0], -v[2, 0]])

ax.set_zlim3d(-rg, rg)

