# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:01:00 2016

@author: dean
"""

import os
import sys
import numpy as np
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)
import d_img_process as imp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import measure
from skimage import filters
#first get an imageset
#img_dir = top_dir + 'images/baseimgs/PC370/'
#stack, stack_descriptor_dict = imp.load_npy_img_dirs_into_stack(img_dir)
#im = stack[0,:,:]
im = np.load('/Users/deanpospisil/Desktop/net_code/images/baseimgs/PC370/10.npy')
#get the gradient of the image

im = imp.fft_resample_img(im, im.shape[1]*1, std_cut_off = None)
n = filters.sobel(im)
d = filters.sobel(n)





#l=measure.find_contours(n, 1.93, fully_connected='low', positive_orientation='low')
#l= np.array(l)
#plt.plot(l[0][:,0], l[0][:,1])

#plt.imshow(d, cmap = cm.Greys_r, interpolation = 'none')
#then take its max

def neigbor_max_2d(im, cur_ind, dni_ind):
    cur_ind = np.array(cur_ind)
    dni_ind = np.array(dni_ind)
    around = np.array([[1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    aind = [cur_ind + shift for shift in around 
            if ((cur_ind + shift)>=0).any() and not ((cur_ind + shift) == dni_ind).any() ]
    max_ind = aind[np.argmin([im[i[0], i[1]] for i in aind])]    
    return max_ind

line = []
first_peak = np.unravel_index(np.argmin(d), d.shape)
cur_peak = neigbor_max_2d(d, first_peak, (1,1))
line.append(first_peak)
line.append(cur_peak)
i = 0
while not (line[0]==cur_peak).all():
    cur_peak = neigbor_max_2d(d, line[i+1], line[i])
    i+=1
    line.append(cur_peak)

line = np.array(line)

#
#then always step to the next higest pixel that was not the previous pixel
#
#until you get to the original pixel