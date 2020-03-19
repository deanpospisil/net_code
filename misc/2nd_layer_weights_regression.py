# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:09:52 2016

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:45:44 2016

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


layer = 0
sample_rate_mult = 10
ims = afile[layer][1]

ims = np.array([im for im in ims])
ims = np.sum(ims, 1)[:48, ...]
ims = ims - np.mean(ims,axis =(1,2), keepdims=True)
fims = np.abs(np.fft.fft2(ims, s=np.array(np.shape(ims)[1:])*sample_rate_mult))
fims_or_index = np.max(fims, axis=(1,2))/np.sum(fims, axis=(1,2))
oriented_index = fims_or_index>np.percentile(fims_or_index, 20)
fims = fims[oriented_index, ...]

'''
fims = fims.reshape(np.shape(fims)[0], (11*sample_rate_mult)**2)
c = get2dCfIndex(11*sample_rate_mult, 11*sample_rate_mult, 11*sample_rate_mult)
mag = np.abs(c).ravel()
ang = np.angle(c)
ang = ang.ravel()
fims=fims[:, ang>0]
ang = ang[ang>0]
ors = np.squeeze((ang[np.argmax(fims, axis=1)] - np.pi/2)%np.pi)

#plt.figure()
#plt.imshow(np.rad2deg(np.fft.fftshift(ang)), interpolation='nearest', cmap=cm.Greys_r)
ims_2 = afile[layer+1][1][:128, oriented_index,...]
ims_2 = np.swapaxes(ims_2, 1, 3)
layer2weights = xr.DataArray(ims_2, dims=['l2', 'r', 'c', 'l1'])
layer2predictors = xr.DataArray([np.cos(ors) + 1j*np.sin(ors)], dims=['p','l1'])
layer2predictors = layer2predictors / (layer2predictors**2).sum('l1')**0.5
fits = (layer2predictors*layer2weights).sum('l1')/((layer2weights**2).sum('l1')**0.5)



unrav_over_last = (np.product(np.shape(ims_2)[:-1]), np.shape(ims_2)[-1])
b = np.reshape(ims_2, unrav_over_last)

ors = np.squeeze(ors)
sorsi = np.argsort(ors)
ors = ors[ sorsi]
b = b[:, sorsi]

freq=2
freq=2
predictor = np.array([np.cos(freq*ors), np.sin(freq*ors)]).T
predictor = predictor / np.sqrt(np.sum(predictor**2, axis=0, keepdims=True))

x, res, ran, s = np.linalg.lstsq(predictor, b.T)
per_var = res/np.sum(b**2, axis=1)
res = res.reshape(ims_2.shape[:-1])
per_var_kern = np.sum(res, axis=(1,2)) / np.sum(ims_2**2, axis=(1,2,3))

cor = np.sqrt(1-per_var)
recon = np.dot(predictor, x).T

plt.subplot(211)
plt.stem(np.sqrt(1-per_var_kern))
plt.title('2nd layer fits (fitting only '+ str(ims_2.shape[-1]) +
                ' top oriented 1st layer kernels)')

plt.ylabel('r')
plt.xlabel('Second Layer Kernel')
plt.ylim(0,1)
plt.tight_layout()


plt.subplot(212)
bf = np.argmax(cor)
loc = np.unravel_index(bf, ims_2.shape[:-1]) 
plt.plot(np.rad2deg(ors), recon[bf,:])
plt.scatter(np.rad2deg(ors), b[bf,:])
plt.ylabel('kernel weight')
plt.xlabel('Orientation (degrees)')
plt.xlim(0,180)
plt.title('best fit kernel pixel r = ' +str(np.round(cor[bf], decimals=2)))
plt.tight_layout()

#plt.subplot(313)
#recon_orig = np.reshape(recon, ims_2.shape[:])
#recon_orig = recon_orig[31,...].reshape(25, 38)
#
#b_orig = np.reshape(b, ims_2.shape[:])
#b_orig = b_orig[31,...].reshape(25, 38)
#
#for ind in range(1):
#    plt.plot(np.rad2deg(ors), recon_orig[ind,:])
#    plt.scatter(np.rad2deg(ors), b_orig[ind,:])
    
    


print(loc)
print(cor[bf])
print(my_cor(recon[bf,:], b[bf,:]))

cor_im = np.reshape(cor, ims_2.shape[:-1])
plt.figure()
data = vis_square(cor_im, padsize=2, padval=0)
plt.xticks([])
plt.yticks([])
plt.title('Plaid-Preference Model r Map')

x, res, ran, s = map(np.array, zip(*[np.linalg.lstsq(predictor, b.T) 
                for ind in range(10)]))


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