# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:45:44 2016

@author: deanpospisil
"""

import sys, os
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
sys.path.append(top_dir + 'nets')


import d_misc as dm
import d_img_process as imp




if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f) 
layer = 0
sample_rate_mult = 2
ims = a[layer][1]

ims = np.array([im for im in ims])
ims = np.sum(ims, 1)
fims = np.abs(np.fft.fft2(ims))
pims = imp.cart_to_polar_2d_lin_broad(fims, sample_rate_mult)
angles = imp.cart_to_polar_2d_angles(11, sample_rate_mult)

ors = (angles[np.argmax(
      np.sum(pims, axis=1, keepdims=True), axis=2)]
      + np.pi/2)%2*np.pi
ors = np.squeeze(ors)[:48]
predictor = np.array([np.ones(np.shape(ors)), np.cos(ors), np.sin(ors)])
predictor = np.array([ np.cos(ors), np.sin(ors)])

predictor = predictor / np.sum(predictor**2,0, keepdims=True)
ims_2 = a[layer+1][1]
resper = []
xs=[]
for ind in range(48):
    b = ims_2[ind,:, 2,2]
    
    x,res,ran,s = np.linalg.lstsq(predictor.T, b)
    xs.append(x)
    resper.append(res/(np.linalg.norm(b)**2))
resper = np.sqrt(1-np.array(resper))
plt.close('all')
plt.subplot(211)
plt.plot(resper)

bf = np.argmax(resper)
sorsi = np.argsort(ors)
plt.subplot(212)

plt.scatter(ors[sorsi], ims_2[bf,sorsi,3,3])
plt.plot(ors[sorsi], (np.dot(np.expand_dims(xs[bf],0), predictor)).T[sorsi], color='r')