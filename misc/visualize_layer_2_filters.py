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

if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)
layer = 1  

for plot_ind, ind in enumerate([0,]):  
    ims = a[layer][1][ind:ind+64]
    
    
    kernstack = ims[0]
    first = True
    for kernstack in ims:
        
        kerns=np.reshape(kernstack, ((np.product(np.shape(kernstack)[:-1]),) 
                                + (np.shape(kernstack)[-1],))) 
        kerns = np.insert(kerns, slice(np.shape(kernstack)[-1], np.product(np.shape(kernstack)[:-1]),np.shape(kernstack)[-1]), 
                                    values=0*np.ones((1,np.shape(kernstack)[-1])),axis=0)
        kerns = kerns/np.max(np.abs(kerns))
        if first == True:
            first = False
            full_im = kerns
        else:
            full_im = np.hstack((full_im, 0*np.ones((np.shape(kerns)[0],1))))
            full_im = np.hstack((full_im, kerns))
    #plt.subplot(2,1,plot_ind+1)
    #max_abs = np.max(np.abs(ims))
    #plt.imshow(full_im, interpolation='nearest', cmap= plt.cm.RdBu, vmin=-max_abs/3, vmax=max_abs/3)  
    plt.imshow(full_im, interpolation='nearest', cmap=plt.cm.PRGn, vmin=-1, vmax=1)  
#    plt.yticks(list(range(0,6*48,12)))
#    plt.gca().set_yticklabels(list(range(0,48,2)))
#    plt.xticks(list(range(0,6*128, 12)))
#    plt.gca().set_xticklabels(list(range(0,128,2)))
    
    plt.xlabel('2nd layer outputs (' + str(ind+1) + ' : '+str(ind+128)+')')
    plt.ylabel('1st layer inputs (48)')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('PRGn_layer_2.eps', format='eps', dpi=500)

