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
from skimage import filters
import d_curve as dc

def pixel_arc(pix_ref, pix_n, radius, arclen, npoints):
    cmplx = np.array([1, 1j])
    pix_dir = pix_ref - pix_n
    ang = np.angle(np.sum(pix_dir*cmplx))

    shifts = np.exp(np.linspace(-arclen/2, arclen/2, npoints)*1j)
    center = np.exp(ang*1j) #rotate by 180 degrees
    cpoints = radius*(shifts*center)
    rpoints = np.round(np.array([np.real(cpoints), np.imag(cpoints)]).T)
    return rpoints

def arc_neigbor_max_2d(im, cur_ind, around):
    cur_ind = np.array(cur_ind)
    aind = [cur_ind + shift for shift in around
            if ((cur_ind + shift)>=0).any()  ]
    min_ind = aind[np.argmax([im[i[0], i[1]] for i in aind])]
    return min_ind

def trace_edge(im, scale, radius, arclen, npts = 100, maxlen = 1000):

    #resample image
    ims = imp.fft_resample_img(im, im.shape[1]*scale, std_cut_off = None)
    temp = np.gradient(ims)#get the gradient of the image
    d = temp[0]**2 + temp[1]**2

    #start at first peak
    first_peak = np.array(np.unravel_index(np.argmax(d), d.shape))
    around = pixel_arc(first_peak, first_peak, radius, np.pi*2, npts)
    cur_peak = arc_neigbor_max_2d(d, first_peak, around)


    line = []
    line.append(first_peak)
    line.append(cur_peak)
    i = 0
    #append to line, till too long, or wraps around
    while not (line[0]==cur_peak).all() and len(line)<maxlen:
        around = pixel_arc(line[i+1], line[i], radius, arclen, npts)
        cur_peak = arc_neigbor_max_2d(d, line[i+1], around)
        i+=1
        line.append(cur_peak)
        if np.sum(np.array(line[0]-cur_peak)**2)**0.5<(radius-1):
            break

    return np.array(line), d

#first get an imageset
#img_dir = top_dir + 'images/baseimgs/PC370/'
#stack, stack_descriptor_dict = imp.load_npy_img_dirs_into_stack(img_dir)
#im = stack[0,:,:]
im = np.load('/Users/dean/Desktop/Modules/net_code/images/baseimgs/formlet/1.npy')
scale = 1.
radius = 1.

effective_radius = radius/scale
arc_len = 2*np.arcsin(effective_radius)
print(np.rad2deg(arc_len))

line,d = trace_edge(im, scale, radius, arc_len, npts = 100, maxlen = 2000)
cmplx = np.array([1, 1j])


#kernlen = 50
#kernel = np.fft.fft(np.array(np.ones(kernlen)), len(line[:,0]))
#line[:,0] = np.fft.ifft(np.fft.fft(line[:,0])*kernel)
#line[:,1] = np.fft.ifft(np.fft.fft(line[:,1])*kernel)

cshape = line*cmplx
cshape= np.sum(cshape,1)

from scipy import interpolate
x = np.real(cshape)
y = np.imag(cshape)
tck,u = interpolate.splprep([x,y], s=0)
unew = np.arange(0, 1.001, 0.01)
out = interpolate.splev(unew, tck)
cshape=out[0]+1j*out[1]


direction = dc.curveOrientations(cshape)
direction = dc.curveAngularPos(cshape)
magnitude = dc.curveCurvature(cshape)



comb=(magnitude)*direction
comb = comb[2:-3]
cshape = cshape[2:-3]
plt.quiver(np.real(cshape),np.imag(cshape),np.real(comb),np.imag(comb),width=0.001,)
plt.plot(np.real(cshape),np.imag(cshape))
plt.axis('equal')
'''
plt.subplot(2,1,1)
plt.imshow(im, cmap = cm.Greys_r, interpolation = 'none')
plt.scatter(line[:,1]/scale, line[:,0]/scale)
plt.plot(line[:,1]/scale, line[:,0]/scale)
plt.axis('equal')
plt.subplot(2,1,2)
plt.imshow(d, cmap = cm.Greys_r, interpolation = 'none')
plt.scatter(line[:,1], line[:,0])
plt.plot(line[:,1], line[:,0])
plt.axis('equal')

kernlen = 20
kernel = np.fft.fft(np.array(np.ones(kernlen)), len(line[:,0]))
c_line = np.fft.ifft(np.fft.fft(line[:,0])*kernel)
v_line = np.fft.ifft(np.fft.fft(line[:,1])*kernel)
plt.scatter(v_line/kernlen, c_line/kernlen)
plt.plot(line[:,1], line[:,0])
'''