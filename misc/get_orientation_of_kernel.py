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
plt.close('all')

import d_misc as dm
import d_img_process as imp
import pandas as pd



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


layer = 0
sample_rate_mult = 10
ims = afile[layer][1]

ims = np.array([im for im in ims])
ims = np.sum(ims, 1)
ims = ims - np.mean(ims,axis =(1,2), keepdims=True)
fims = np.abs(np.fft.fft2(ims, s=np.array(np.shape(ims)[1:])*sample_rate_mult))
fims = fims.reshape(96, (11*sample_rate_mult)**2)
c = get2dCfIndex(11*sample_rate_mult, 11*sample_rate_mult, 11*sample_rate_mult)
mag = np.abs(c).ravel()
ang = np.angle(c)
ang = ang.ravel()
fims=fims[:, ang>0]
ang = ang[ang>0]
ors = (ang[np.argmax(fims, axis=1)] - np.pi/2)%np.pi

#plt.figure()
#plt.imshow(np.rad2deg(np.fft.fftshift(ang)), interpolation='nearest', cmap=cm.Greys_r)
ims_2 = afile[layer+1][1]
ims_2 = np.swapaxes(ims_2, 1, 3)
unrav_over_last = (np.product(np.shape(ims_2)[:-1]), np.shape(ims_2)[-1])
b = np.reshape(ims_2, unrav_over_last)


ors = np.squeeze(ors)[:48]
sorsi = np.argsort(ors)
ors = ors[ sorsi]
b = b[:, sorsi]


predictor = np.array([ np.cos(2*ors), np.sin(2*ors)]).T
x,res,ran,s = np.linalg.lstsq(predictor, b.T)

recon = np.dot(predictor, x).T

plt.scatter(b[100,:], recon[100,:])
recon = recon.reshape(np.shape(ims_2))

'''
plt.figure()
data = ims
n = int(np.ceil(np.sqrt(data.shape[0])))
data = (data - data.min()) / (data.max() - data.min())

#for ind in range(len(data)):
#    plt.subplot(10, 10,ind+1)
#    _ = (data[ind] - data[ind].min()) / (data[ind].max() - data[ind].min())
#    plt.imshow(_, interpolation='None',cmap=cm.Greys_r )
#    plt.gca().set_xticks([])
#    plt.gca().set_yticks([])
#    plt.title(str(ind) + ': ' + str(np.round(np.rad2deg(ors[ind]))))
#plt.tight_layout()

ors = np.squeeze(ors)[:48]
predictor = np.array([ np.cos(2*ors), np.sin(2*ors)])

predictor = predictor / np.sum(predictor**2, 0, keepdims=True)
ims_2 = afile[layer+1][1]
resper = []
xs=[]

for ind in range(128):
    b = ims_2[ind,:, 2, 2]
    x,res,ran,s = np.linalg.lstsq(predictor.T, b)
    xs.append(x)
    resper.append(res/(np.linalg.norm(b)**2))


resper = np.sqrt(1-np.array(resper))
plt.figure()
plt.subplot(211)
plt.plot(resper)
plt.ylabel('r')
plt.xlabel('Second Layer Kernel Ind')
plt.tight_layout()

bf = np.argmax(resper)
sorsi = np.argsort(ors)
plt.subplot(212)



plt.scatter(np.rad2deg(ors[sorsi]), ims_2[bf,sorsi,2,2])
plt.plot(np.rad2deg(ors[sorsi]), (np.dot(np.expand_dims(xs[bf],0), predictor)).T[sorsi], color='r')
plt.ylabel('kernel weight')
plt.xlabel('Orientation (degrees)')
plt.xlim(0,180)
plt.title('best fit kernel: ' + str(bf))
plt.tight_layout()
'''