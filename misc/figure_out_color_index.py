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
    
plt.close('all')
# convert these to circular
#for cind in [55,56,82,95,60,59,94]:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.figure()
layer = 0    
ims = a[layer][1]
ims = np.array([im for im in ims])
low_rank = []
q,r=np.linalg.qr(np.array([[1,1,1],[1,1,0],[0,1,1]]).T)
count=0
for cind in range(95):

    im = ims[cind,:,:]
    im = np.swapaxes(im, 0,2)
    #im = im - np.mean(im)
    #im = im / np.linalg.norm(im)
    im = im.astype(np.float64)
    #plt.close('all')
    #plt.imshow((im - im.min()) / (im.max() - im.min()), interpolation='nearest')
    
    tot_mag = np.linalg.norm(im)**2
    white_comp = np.tile(np.mean(im, 2, keepdims=True), (1, 1, 3))
    color_comp = im - white_comp
    
    #print(np.isclose(im, (white_comp+color_comp)).all())
    #color_comp = color_comp.T
    #plt.imshow((color_comp - color_comp.min()) / (color_comp.max() - color_comp.min()), interpolation='nearest')
    
    #print(np.linalg.norm(white_comp)**2)
    #print(np.linalg.norm(color_comp)**2)
    #print(np.linalg.norm(white_comp)**2 + np.linalg.norm(color_comp)**2)
    #print(tot_mag)
    

    rgb = np.reshape(im, (11*11, 3))
#    ax.scatter(rgb[:, 0], 
#               rgb[:, 1], 
#               rgb[:, 2])
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    
    rg = np.max(np.abs(im))
    ax.set_xlim3d(-rg, rg)
    ax.set_ylim3d(-rg, rg)
    

    cov = np.dot(rgb.T, rgb)
    w, v = np.linalg.eig(cov)
    
    v=v*0.2*(max(w)/sum(w))
    if max(w)/sum(w)>0.95:
        count+=1
        print(count)
        print(max(w))
        low_rank.append(im)
        ax.plot(xs=[v[0, 0], -v[0, 0]], ys=[v[1, 0], -v[1, 0]], zs=[v[2, 0], -v[2, 0]])
    
    ax.set_zlim3d(-rg, rg)
    
    #plane orthogonal
    
    chrom_coords = np.dot(rgb, q[:,1:])
    rgb = rgb -np.min(rgb)
    rgb = rgb / np.max(rgb)
    if frac_var_chrom(im)>0.5:
        plt.scatter(x=chrom_coords[:,0].T, y=chrom_coords[:,1], s=0.2, 
                color=rgb)