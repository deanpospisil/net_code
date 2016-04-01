# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:51:00 2016

@author: deanpospisil
"""

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

import itertools
flatten_iter = itertools.chain.from_iterable
def factors(n):
    return set(flatten_iter((i, n//i) 
                for i in range(1, int(n**0.5)+1) if n % i == 0))
        
if 'a' not in locals():
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
def big_square(n):
    len_clust = n
    f = np.sort(list(factors(len_clust)))
    f1 = f
    f2 = len_clust/f1
    fact1 =f1[np.argmin(np.abs(f1-f2))]
    fact2 =f2[np.argmin(np.abs(f1-f2))]
    return (fact1, fact2)
    

layer = 0    
ims = a[layer][1]
ims = np.array([im for im in ims])
ims = np.swapaxes(ims, 1, 3)
ims = ims.reshape(np.product(np.shape(ims)[:-1]), np.shape(ims)[-1])
ims_n = ims/np.sum(ims**2, 1, keepdims=True)**0.5
cor = np.dot(ims_n, ims_n.T)
thresh = 0.95
clust_ims = []
n_clusts = 10
for nclust in range(n_clusts):
    boolind = cor>thresh
    proto = np.argmax(np.sum(boolind, 1))
    boolind = boolind[proto, :]
    cluster = ims[boolind, :]
    len_clust = np.sum(boolind)
    
    big_shape = big_square(len_clust)
    side = int((len_clust)**0.5)
    big_shape = (side, side)
    cluster = cluster[:side**2, :]
    
    
    clust_im = np.reshape(cluster, (big_shape+(3,)))
    clust_im = (clust_im - clust_im.min()) / (clust_im.max() - clust_im.min())
    clust_ims.append(clust_im)
    cor = cor[-boolind,:]
    cor = cor[:,-boolind]
    ims = ims[-boolind,:]
    plt.subplot(1, n_clusts, nclust+1)
    plt.imshow(clust_im, interpolation = 'nearest')
 
print('Clusters account for:' + str(1 - ims.shape[0]/ims_n.shape[0]))






'''
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
    plt.title(str(ind) +': '+ str(np.round(fc, decimals=3)))
plt.tight_layout()
'''