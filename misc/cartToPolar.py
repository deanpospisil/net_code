# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:53:48 2015

@author: dean
"""


#test for non-square images.
import matplotlib.cm as cm
import numpy as np
import numpy.fft as f
import matplotlib.pyplot as p
pi=np.pi

def getfIndex(nSamps, fs):
    
    f = np.fft.fftfreq(int(nSamps),1./fs)

    return f

def get2dCfIndex(xsamps,ysamps,fs):
    fx, fy = np.meshgrid( getfIndex(xsamps,fs), getfIndex(ysamps,fs) )
    C = fx + 1j * fy
    return C


def fft2Interpolate(coef, x, y, w):
    intrpvals = np.zeros( np.size(x), 'complex')
    #vectorize this later
    basis = np.exp( 1j * 2 * pi *  (x * w1 + y * w2 ) )
    intrpvals = np.sum(  coef * basis, 1 )

    return intrpvals
    
#here is my original image that I will then get new samples for
n = 50.
s=(np.arange(n))/n
x, y = np.meshgrid(s,s)

#lets put in two sinusoids, I am not going to bother with symmetry, I canjust multiply by two ate the end and its the same
wave=1j*np.ones(np.size(x))

#xm = 4 * np.cos(1) 
#ym = 4 * np.sin(1) 

xm =1
ym=1
lowwave = np.real(np.exp(  1j * 2 * pi * ( x * xm + y * ym ) ) )

#get its fourier coefficients
ftl = f.fft2(lowwave)
amp = abs(ftl) / np.size(ftl)#scale to length of signal, so these terms are the amplitude of the sinusoids
phase = np.exp(1j*np.angle(ftl) )
coef = amp * phase

#now I am going to choose new positions to evaluate this fft at, I happen to be choosing
#evenly spaced ones, but I can choose any arbitrary values.
newN = 100.
s=(np.arange(newN))/newN
x, y = np.meshgrid(s,s)
x = x.ravel()
y = y.ravel()


#get the frequency index, with the sampling rate as the number of samples of the original image
#this way the spatial frequencies are the same in the original and upsampled image, with respect to image boundaries not pixels
#this is all identical to zero padding in the frequency domain.
tmp = get2dCfIndex(n, n, n)

w1= np.real(tmp).ravel()
w2 = np.imag(tmp).ravel()
coef = coef.ravel()

intrpvals = np.zeros( np.size(x), 'complex')
#vectorize this later


highwave = fft2Interpolate( coef, x, y , w1 ) 

#put them back into a square, so we can plot it
highwave.shape = ( newN, newN )
p.clf()

p.subplot(221)
p.imshow(np.real(lowwave), interpolation='None',cmap = cm.Greys_r)#just get real part because I didn't bother with symmetry
p.subplot(222)
p.imshow(np.abs(f.fftshift(f.fft2(lowwave))), interpolation='None',cmap = cm.Greys_r)

#note no new spatial frequencies have been created, you've just sampled the same frequencies more often
p.subplot(223)
p.imshow(np.abs(highwave), interpolation='None',cmap = cm.Greys_r)
p.subplot(224)
p.imshow(np.abs(f.fftshift(f.fft2(highwave))), interpolation='None',cmap = cm.Greys_r)