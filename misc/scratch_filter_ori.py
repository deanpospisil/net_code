# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:13:23 2017

@author: deanpospisil
"""
import numpy as np
import matplotlib.pyplot as plt 
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

    if img.ndim == 3:
        polar_img = img[Ycart, Xcart, :]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img

a_filter = np.zeros((11,11,3))
a_filter[:,3,:] = 1

a_filter_unwrap = a_filter.reshape(np.product(a_filter.shape[:-1]), 3)#npixelsX3
u, s, v = np.linalg.svd(a_filter_unwrap, full_matrices=False)#highest var dir
#get covariance with the highest dir of var
val_map = np.dot(a_filter_unwrap, v[0,:]).reshape(np.shape(a_filter)[:-1])
val_map = np.array(val_map)

sample_rate_mult = 20
s=tuple(np.array(np.shape(a_filter)[:2])*sample_rate_mult)
upsampled_fft_amplitude = np.abs(np.fft.fft2(val_map,s=s))
radius = np.floor(upsampled_fft_amplitude.shape[0]/2)
polar = img2polar(np.fft.fftshift(upsampled_fft_amplitude), [radius,radius], radius, phase_width=360)
or_dist =polar.sum(0)

plt.plot(or_dist)
                   #%%
ppolar_amp_kurtosis = np.array([coef_var(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_deg = np.array([np.argmax(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_rad = np.deg2rad(prfrd_ori_deg)
prfrd_ori_rad_wrp = prfrd_ori_rad%np.pi