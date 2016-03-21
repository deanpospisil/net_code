# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:12:06 2016

@author: deanpospisil
"""

import pickle
import os, sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')
import xarray as xr



with open(top_dir + 'nets/netwts.p', 'rb') as f:
    
    a = pickle.load(f, encoding='latin1')

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

    return im_pol

def cart_to_polar_2d_lin_broad(im, sample_rate_mult):
    cut = [cart_to_polar_2d_lin(im_cut, sample_rate_mult) for im_cut in  im]
    return cut

def circ_cor(pol1, pol2, sample_rate_mult=2):


    cross_cor = np.fft.ifft(np.fft.fft(pol1, axis=1) *
                        np.fft.fft(np.fliplr(pol2), axis=1),
                        axis=1)
    sum_over_r = np.sum(np.real(cross_cor), axis=0)

    return sum_over_r, angles

# convert these to circular
ims = a[0][1]
ims = np.array([im for im in ims])


sample_rate_mult = 1
pol_ims = np.array([cart_to_polar_2d_lin_broad(im, sample_rate_mult) for im in ims])
pol_ims = pol_ims / np.sqrt(np.sum(pol_ims**2, axis=(1,2,3), keepdims=True))
pol_ims = np.fft.fft(pol_ims)

p3 = np.expand_dims(pol_ims, 0) * np.conj(np.expand_dims(pol_ims, 1))

p4 = np.fft.ifft(p3)
p4 = np.sum(p4, axis = (2, 3))
cormat=np.real(np.max(abs(p4), 2))
#plt.plot(np.diag(cormat))
plt.figure()
plt.imshow(np.real(np.max(abs(p4), 2)), interpolation='None', cmap=cm.Greys_r)
plt.colorbar()
plt.title('rotational correlation, AlexNet Kernels Layer 1')
plt.xlabel('unit #')
plt.ylabel('unit #')


w, v =np.linalg.eig(cormat)
data = ims
n = int(np.ceil(np.sqrt(data.shape[0])))
data = (data - data.min()) / (data.max() - data.min())

for ind in range(len(pol_ims)):
    plt.subplot(10, 10,ind+1)
    plt.imshow( np.swapaxes(data[ind],0,2), interpolation='None')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(str(ind))
plt.tight_layout()