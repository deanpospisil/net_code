# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:15:33 2015

@author: dean
"""

import downSampling as ds
from PIL import Image, ImageFilter
from scipy import interpolate
import matplotlib.pyplot as p
import formlet
#import Image
import numpy as np
pi=np.pi

def fft2Interpolate(coef, points, w):
    
    basis = np.exp( 1j * 2 * pi * ( points[0,0] * w[0] + points[0,0] * w[1] ))
    nPoints = np.size(points)/2
    intrpvals = np.zeros( nPoints, 'complex')
    for ind in xrange(nPoints):
        basis[:,:] = np.exp( 1j * 2 * pi * ( points[0, ind] * w[0] + points[1, ind] * w[1] ) )
        intrpvals[ind] = np.sum(  coef * basis  )
    return intrpvals

def circPoints(center, radius, theta):
    circ = center[:,None] + radius * np.array([np.cos(theta), np.sin(theta)])
    return circ 

nPts=100
center =np.array([1,1])
radius= 1
theta = np.linspace(0.0, 2*pi-2*pi/nPts, nPts)
points = circPoints(center, radius, theta)
p.plot(points[0], points[1])
p.show()
#
#def get2ndHarmonicMin(center, radius, w , coef):
#    nPts = 9
#    angles = np.linspace(0.0, 2*pi-2*pi/nPts, nPts)
#    
#    points = circPoints(center, radius, angles)
#    trps = fft2Interpolate(coef, points, w)
#    
#    ft=np.fft.fft(trps)
#    minTheta = np.angle(ft[2]) - pi/2
#    
#    center1 = circPoints(center, radius, minTheta)
#    center2 = circPoints(center, radius, minTheta - pi)
#    return center1, center2
#
#def boundaryTrace2Harm(initCenter, radius , w, coef ):
#    center1, center2 = get2ndHarmonicMin(initCenter, radius,w, coef)
#    bPoints = [initCenter]
#    bPoints.append(center1)
#    newCenter = np.array([np.nan, np.nan])
#    
#    while radius < np.linalg.norm(newCenter-initCenter) and len(bPoints)<100000:
#        
#        center = bPoints[-1]        
#        center1, center2 = get2ndHarmonicMin( center, radius,w, coef )
#        
#        if np.linalg.norm(center1 - bPoints[-2]) > np.linalg.norm(center1 - bPoints[-2]):
#            newCenter = center1
#        else:
#            newCenter = center2
#        bPoints.append(newCenter)
#    return bPoints           
        
    
#def boundaryCircularGD( gain, center, radius, d1, coef, w):
#    init = 0
#    x, y = circPoints(center, radius, [init + d1, init]) 
#    d2 = 1
#    
#    while np.diff(trps) > 1*10**-16 and d2 > 1*10**-16  :
#        
#        trps = fft2Interpolate(coef, x, y, w)
#        d2 = d1 - gain * np.diff(trps)/delta
#        init = 
#        x, y = circPoints(center, radius, [ init + d2, init ])  
#        
#    return minVal


#
#
##make the initial complex circle
#nPts=10000
#angles = np.linspace(0.0, 2*pi-2*pi/nPts, nPts)
#radius = 0.15
#cShape = np.exp(angles*1j)*radius
#startSigma=0.3
#endSigma=0.1
#nFormlets=32
#meanFormDir = np.deg2rad(270) 
#stdFormDir = np.deg2rad(70)
#meanFormDist = radius
#stdFormDist = 0
#
##make the evenly sampled shape
#theShape, x , y,  sigma, alpha = formlet.makeNaturalFormlet(nPts, radius, nFormlets ,meanFormDir, stdFormDir, meanFormDist, stdFormDist, startSigma, endSigma )
#tck,u = interpolate.splprep([x,y], s=0)
#unew = np.arange(0, 1.00, 0.0001)
#out = interpolate.splev(unew, tck) 
#theShape=out[0]+1j*out[1]
#magO = formlet.getComplexJordanCurveCurvature(theShape)
#
#
##save the image
#p.figure(figsize=(10,10))#this is roughly resolution/96
#points = np.array([x, y]).T
#line = p.Polygon(points, closed=True, fill='k', edgecolor='none',fc='k')
#p.gca().add_patch(line)
#p.axis('off')
#p.gca().set_xlim([-radius*2, radius*2])
#p.gca().set_ylim([-radius*2, radius*2])
#p.savefig('shape.png',bbox_inches=None)
#img = Image.open('shape.png')
#ima = np.array(img)[:,:,0]#load the image
#p.clf()
#
#dsRate = 10.
#imSize = np.shape(ima)[0]
#dsImSize = imSize/dsRate
#img = ima
#
#fT = np.fft.fft2(img)
#dR = dsImSize
#dC = dR
#
#
#fs=imSize
#fIndex =ds.get2dCfIndex( imSize, imSize, fs  )
#c = abs(fIndex)
#
#oldSize =imSize
#newSize=dsImSize
#stdCutOff = 4
#stdFreq = ds.guassianDownSampleSTD( oldSize, newSize, stdCutOff, fs )
#
#f = abs( ds.get2dCfIndex( imSize, imSize, fs) )
#filt = ds.myGuassian( f, 0, stdFreq )
#
#fT=fT*filt
#convImg = np.fft.ifft2(fT)
#dConvImg = np.gradient(convImg)
#dConvImg = np.real(np.sqrt(dConvImg[0]**2+dConvImg[1]**2))
#start = np.unravel_index(np.argmax(dConvImg),np.shape(dConvImg))
#start = np.array(start)/np.double(imSize)
#fi=img.filter(ImageFilter.CONTOUR)
#
#
##
##d2ConvImg = np.gradient(dConvImg)
##d2ConvImg = np.real(np.sqrt(d2ConvImg[0]**2+d2ConvImg[1]**2))
##
##ftl = np.fft.fft2(d2ConvImg)
##amp = abs(ftl) / np.size(ftl)
##phase = np.exp(1j*np.angle(ftl) )
##coef = amp * phase
##
###go through each x, y
##tmp = ds.get2dCfIndex(imSize, imSize, imSize)
##w=np.zeros( (2,imSize,imSize) )
##w[0] = np.real(tmp)
##w[1] = np.imag(tmp)
##
##radius = (dsRate * imSize**-1) / 4
##points = boundaryTrace2Harm(start, radius , w, coef )
##print(points)
#
#p.clf()
#p.subplot(211)
#p.imshow(np.real(ima), interpolation='None',cmap = cm.Greys_r)
#p.subplot(212)
#p.imshow(d2ConvImg, interpolation='None',cmap = cm.Greys_r)
#
#
