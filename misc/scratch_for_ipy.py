# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 11:47:07 2016

@author: deanpospisil
"""

import sys, os
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
sys.path.append(top_dir + 'nets')
plt.close('all')

import xarray as xr
def polar2cart(r, theta, center):

    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center, final_radius, initial_radius = None, phase_width = 3000):

    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img
    
def my_cor(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    r = np.dot(a, b)
    return r
def vis_square(data, padsize=0, padval=0):

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='nearest', cmap = cm.hot, vmin=0, vmax=1)
    plt.colorbar()

    plt.tight_layout()
    return data


def get2dCfIndex(xsamps, ysamps,fs):
    fx, fy = np.meshgrid(np.fft.fftfreq(int(xsamps),1./fs),
                         np.fft.fftfreq(int(ysamps),1./fs) )
    c = fx + 1j * fy
    return c


if 'afile' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            afile = pickle.load(f, encoding='latin1')
        except:
            afile = pickle.load(f)
def coef_var(a):
    mu = a.mean()
    sig = a.std()
    return 1./(((sig/mu)**2)+1)
    
def sinusoid_weights_test(orientation_of_inputs, weights_on_outputs, freq=2):
    lyr_2_prd_df = xr.DataArray([np.cos(freq*orientation_of_inputs.values), 
                                 np.sin(freq*orientation_of_inputs.values)], 
                                dims=['p','l1'])
    lyr_2_prd_df_nrm = lyr_2_prd_df / (lyr_2_prd_df**2).sum('l1')**0.5
    
    fits = (lyr_2_prd_df_nrm*lyr_2_wts_df).sum('l1').squeeze()
    lyr_2_wts_df_hat = (fits * lyr_2_prd_df_nrm).sum('p')

    lyr_2_wts_df_hat_nrm = lyr_2_wts_df_hat / (lyr_2_wts_df_hat**2).sum(['l1','r','c'])**0.5
    lyr_2_wts_df_nrm = lyr_2_wts_df / (lyr_2_wts_df**2).sum(['l1','r','c'])**0.5
    cor = (lyr_2_wts_df_hat_nrm * lyr_2_wts_df_nrm).sum(['l1','r','c'])
    return cor
    
layer = 0
sample_rate_mult = 10
ims = afile[layer][1]

lyr_1_grp_sze = 48
lyr_2_grp_sze = 128

first_layer_weights = np.array([im for im in ims])


first_layer_weights_grey_scale = np.sum(ims, 1)[:lyr_1_grp_sze, ...]
first_layer_weights_grey_scale -= np.mean(first_layer_weights_grey_scale, axis =(1,2), keepdims=True)
upsampled_fft_amplitude = np.abs(np.fft.fft2(first_layer_weights_grey_scale, 
                        s=np.array(np.shape(first_layer_weights_grey_scale)[1:])*sample_rate_mult))

polar = [img2polar(np.fft.fftshift(a_filter), [55,55], 55, phase_width=360)
                    for a_filter in upsampled_fft_amplitude]
    
polar_amp_kurtosis = np.array([coef_var(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_deg = np.array([np.argmax(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_rad = np.deg2rad(prfrd_ori_deg)
prfrd_ori_rad_wrp = prfrd_ori_rad%np.pi

power_concentration = upsampled_fft_amplitude.max((1,2)) / upsampled_fft_amplitude.sum((1,2))
top_pwr_cncntrtn_ind = polar_amp_kurtosis<np.percentile(polar_amp_kurtosis, 80)
