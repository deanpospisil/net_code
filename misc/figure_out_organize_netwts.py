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
#pol_ims = np.array([di.cart_to_polar_2d_lin_broad(im, sample_rate_mult) for im in ims])

pol_ims = pol_ims - np.mean(pol_ims, axis=(2, 3), keepdims=True)
pol_ims = pol_ims / np.sqrt(np.sum(pol_ims**2, axis=(1,2,3), keepdims=True))
pol_ims = np.fft.fft(pol_ims)
p3 = np.expand_dims(pol_ims, 0) * np.conj(np.expand_dims(pol_ims, 1))

p4 = np.fft.ifft(p3)
p4 = np.real(np.sum(p4, axis=(2, 3)))

plt.close('all')
cormat = p4[:,:,0]
#plt.plot(np.diag(cormat))
plt.figure()
plt.subplot(121)
plt.title('unrotated r Layer ' + str(layer+1))
plt.xlabel('unit #')
plt.ylabel('unit #')
plt.imshow(cormat, interpolation='None', vmin=-1, vmax=1)
ax = plt.gca()
im = ax.imshow(cormat, interpolation='nearest', vmin=-1, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)


cormatrot = np.max(p4, 2)
cormatrot = dm.maxabs(p4, axis=2)
plt.subplot(122)
plt.title(' max abs r rotated')
plt.xlabel('unit #')
plt.ylabel('unit #')
ax = plt.gca()
im = ax.imshow(cormatrot, interpolation='nearest', vmin=-1, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.tight_layout()

#w, v =np.linalg.eig(cormat)
plt.figure()
data = ims
n = int(np.ceil(np.sqrt(data.shape[0])))
data = (data - data.min()) / (data.max() - data.min())

for ind in range(len(pol_ims)):
    plt.subplot(10, 10,ind+1)
    fc = frac_var_chrom(np.swapaxes(data[ind],0,2))
    plt.imshow(np.swapaxes(data[ind],0,2), interpolation='None')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(str(ind) +': '+ str(np.round(fc, decimals=2)))
plt.tight_layout()