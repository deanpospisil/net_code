# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:03:47 2015

@author: dean
"""

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc as mis
import matplotlib.cm as cm

def getfIndex(nSamps, fs):
    
    f = np.fft.fftfreq(int(nSamps),1./fs)

    return f

def get2dCfIndex(xsamps,ysamps,fs):
    fx, fy = np.meshgrid( getfIndex(xsamps,fs), getfIndex(ysamps,fs) )
    C = fx + 1j * fy
    return C


def fft2Interpolate( x, y, w):
    intrpvals = np.zeros( np.size(x), 'complex')
    basis = np.exp( 1j * 2 * np.pi *  (x * w[0] + y * w[1] ) )
    intrpvals = np.sum(  coef * basis, 1, dtype = np.complex128 )

    return intrpvals

os.chdir('/Users/deanpospisil/Desktop/')
im=mpimg.imread('soft_on_orig.png')
im = np.sum(im,2)
ftl = np.fft.fft2(im)
ftl = np.fft.rfft2(im)

amp = abs(ftl) / np.size(ftl)#scale amplitude so these terms are coefficients
phase = np.exp(1j*np.angle(ftl) )
coef = amp * phase

tmp = get2dCfIndex(np.size(im,0), np.size(im,1), np.size(im,0))
w=np.zeros((2,np.size(im,0),np.size(im,1)))
w1 = np.real(tmp)[ :np.size(ftl,0), :np.size(ftl,1) ].ravel()
w2 = np.imag(tmp)[ :np.size(ftl,0), :np.size(ftl,1) ].ravel()

#lets get polar x,ys
npts= 75
angles = np.linspace( 0, 2*np.pi, npts)
magnitudes = np.logspace(np.log10(0.1), np.log10(0.5), npts )
magnitudes = np.linspace(0,.5, npts )


angles, magnitudes = np.meshgrid( angles, magnitudes) 
x = magnitudes * np.cos(angles) + 0.5
y = magnitudes * np.sin(angles) + 0.5

x = x.ravel()
y = y.ravel()

x.shape = (np.size(x,0), 1)
y.shape = (np.size(y,0), 1)

coef = coef.ravel()
w = [w1,w2]
basis = np.exp( 1j * 2 * np.pi *  (x * w[0] + y * w[1] ) )
intrpvals = np.sum( coef * basis, 1, dtype = np.complex128 )
highwave = intrpvals
plt.figure()
#highwave = fft2Interpolate( coef, x, y , [w1,w2] ) 



#put them back into a square, so we can plot it
highwave.shape = ( npts, npts )

plt.subplot(131)
plt.imshow(np.real(im), interpolation='None',cmap = cm.Greys_r)#just get real part because I didn't bother with symmetry
plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.subplot(132)
plt.scatter(x,y)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.subplot(133)
plt.imshow(np.real(highwave), interpolation='None',cmap = cm.Greys_r)
plt.xlabel('Orientation')
plt.ylabel('Log(r)')
plt.gca().set_xticks([])
plt.gca().set_yticks([])




mis.imsave('pol.png', np.real(highwave))

mis.imsave('cart.png', im)
