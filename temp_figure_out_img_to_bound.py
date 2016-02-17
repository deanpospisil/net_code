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
'''
#first get an imageset
#img_dir = top_dir + 'images/baseimgs/PC370/'
#stack, stack_descriptor_dict = imp.load_npy_img_dirs_into_stack(img_dir)
#im = stack[0,:,:]
im = np.load('/Users/dean/Desktop/Modules/net_code/images/baseimgs/PC370/10.npy')
#get the gradient of the image

im = imp.fft_resample_img(im, im.shape[1]*10, std_cut_off = None)
n = np.gradient(im)
d = n[0]**2 + n[1]**2
plt.imshow(d, cmap = cm.Greys_r, interpolation = 'none')
#then take its max

beg = np.unravel_index(np.argmax(d), d.shape)

around = [[1,0], [0,1], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]]
line = []
cur = None
while not beg == cur:
    cur = beg
    last = im[cur]
    aind = [ (shifts[0]+cur[0], shifts[1] + cur[1]) for shifts in around if shifts[0]+cur[0]>=0 and shifts[1]+cur[1]>=0  ]
    beg = aind[ np.argmax([im[i] for i in aind if not im[i]==last])]
    line.append(beg)
'''
pix_ref = np.array([3, 3])
pix_n = np.array([4, 3])
cmplx = np.array([1, 1j])

pix_dir = pix_ref - pix_n

ang = np.angle(np.sum(pix_dir*cmplx))

print np.rad2deg(ang)

radius = 5
arclen = np.pi/2
npoints = 10

def pixel_arc(pix_ref, pix_n, radius, arclen, npoints):
    cmplx = np.array([1, 1j])
    pix_dir = pix_ref - pix_n
    ang = np.angle(np.sum(pix_dir*cmplx))

    shifts = np.exp(np.linspace(-arclen/2, arclen/2, npoints)*1j)
    center = np.exp(ang*1j)*-1 #rotate by 180 degrees
    cpoints = radius*(shifts*center)
    rpoints = np.array([np.real(cpoints), np.imag(cpoints)]).T
    return rpoints




#then always step to the next higest pixel that was not the previous pixel

#until you get to the original pixel