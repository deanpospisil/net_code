# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:20:44 2016

@author: deanpospisil
"""

import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:03:47 2015

@author: dean
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc as mis
import matplotlib.cm as cm
from scipy import interpolate


def get2dCfIndex(xsamps,ysamps,fs):
    fx, fy = np.meshgrid(np.fft.fftfreq(int(xsamps),1./fs),
                         np.fft.fftfreq(int(ysamps),1./fs) )
    c = fx + 1j * fy
    return c

def cart_to_polar_2d_fft(im, sample_rate_mult):
    
    #get polar resampling coordinates
    n_pix = np.size(im,0)/2.
    npts_mag = int(np.size(im,0) / 2.)
    npts_angle = int(np.pi * 2 * n_pix)
    angles_vec = np.linspace(0, 2*np.pi, npts_angle)
    magnitudes = np.linspace(0, 0.5, npts_mag)
    angles, magnitudes = np.meshgrid(angles_vec, magnitudes) 
    x = np.expand_dims((magnitudes * np.cos(angles) + 0.5).ravel(), 1)
    y = np.expand_dims((magnitudes * np.sin(angles) + 0.5).ravel(), 1)
    
    #get fourier coefficients of image    
    ftl = np.fft.rfft2(im)
    amp = abs(ftl) / np.size(ftl)#scale amplitude so these terms are coefficients
    phase = np.exp(1j * np.angle(ftl))#get phase of these coefficients
    coef = (amp * phase).ravel()
    
    #get fourier basis    
    _ = get2dCfIndex(np.size(im,0), np.size(im,1), np.size(im,0))
    w1 = np.real(_)[ :np.size(ftl,0), :np.size(ftl,1) ].ravel()
    w2 = np.imag(_)[ :np.size(ftl,0), :np.size(ftl,1) ].ravel()
    basis = np.exp(1j * 2*np.pi * (x * w1 + y * w2))
    
    #apply coef to basis, and sum
    circ_vals = np.sum(coef * basis, 1, dtype=np.complex128)
    circ_vals.shape = (npts_mag, npts_angle)
    #circ_vals = circ_vals * magnitudes*2*np.pi
    
    return np.real(circ_vals), angles_vec
    
def cart_to_polar_2d_lin(im, sample_rate_mult):
    
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    f = interpolate.interp2d(x, y, im, kind='linear')    
    
    #get polar resampling coordinates
    n_pix = np.size(im,0)/2.
    npts_mag = int(np.size(im, 0) / 2.)*sample_rate_mult
    npts_angle = int(np.pi * 2 * n_pix)*sample_rate_mult
    
    angles_vec = np.linspace(0, 2*np.pi, npts_angle)
    magnitudes_vec = np.linspace(0, n_pix-1, npts_mag)
    angles, magnitudes = np.meshgrid(angles_vec, magnitudes_vec) 
    xnew = (magnitudes * np.cos(angles)+n_pix).ravel()
    ynew = (magnitudes * np.sin(angles)+n_pix).ravel()
    f = interpolate.RegularGridInterpolator((x, y), im, method='linear')
    
    pts = np.fliplr(np.array([xnew, ynew]).T)
    im_pol = f(pts).reshape(npts_mag, npts_angle)
    im_pol = im_pol * magnitudes*2*np.pi
    
    return im_pol, angles_vec

    
os.chdir('/Users/deanpospisil/Desktop/')
im=mpimg.imread('soft_on_orig.png')
im = np.sum(im, 2) 

sample_rate_mult = 2

ft = np.zeros((100,51))
ft[10, 0] = 1
im1 = np.fft.irfft2(ft)
ft = np.zeros((100,51))
ft[0, 10] = 1
im2 = np.fft.irfft2(ft)

plt.subplot(511)
plt.imshow(im1, interpolation='None',cmap = cm.Greys_r)

plt.subplot(512)
plt.imshow(im2, interpolation='None',cmap = cm.Greys_r)

plt.subplot(513)
circ_vals1, angles = cart_to_polar_2d_lin(im1, sample_rate_mult)
#circ_vals1, angles = cart_to_polar_2d_fft(im1, sample_rate_mult)
circ_vals1 = circ_vals1/np.linalg.norm(circ_vals1)


plt.imshow(circ_vals1, interpolation='None',cmap = cm.Greys_r)


plt.subplot(514)
circ_vals2, angles = cart_to_polar_2d_lin(im2, sample_rate_mult)
#circ_vals2, angles = cart_to_polar_2d_fft(im2, sample_rate_mult)

circ_vals2 = circ_vals2/np.linalg.norm(circ_vals2)
plt.imshow(circ_vals2, interpolation='None',cmap = cm.Greys_r)


cross_cor = np.fft.ifft(np.fft.fft(circ_vals1, axis=1) * 
                        np.fft.fft(np.fliplr(circ_vals2), axis=1), 
                        axis=1)
plt.subplot(515)
sum_over_r = np.sum(np.real(cross_cor),axis = 0)
#plt.imshow( np.real(cross_cor), interpolation='None',cmap = cm.Greys_r)
plt.plot(np.rad2deg(angles), sum_over_r)

print(np.rad2deg(angles[np.argmax(sum_over_r)]))

#circ_vals, x, y = cart_to_polar_2d_fft(im)

#plt.figure()
#put them back into a square, so we can plot it

#plt.subplot(121)
#plt.imshow(np.real(im), interpolation='None',cmap = cm.Greys_r)#just get real part because I didn't bother with symmetry
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
#
#plt.scatter((xnew)*im.shape[0], (ynew)*im.shape[0], s=0.1)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
#
#plt.subplot(122)
#plt.imshow(np.real(circ_vals), interpolation='None',cmap = cm.Greys_r)
#plt.xlabel('Orientation')
#plt.ylabel('Lin(r)')
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])

