# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:02:26 2015

@author: dean
"""
import scipy.signal as sig
import scipy
import warnings
import numpy as np
pi=np.pi
import scipy as sc


def saveToPNGDir(directory, fileName, img):
    import os
    if not os.path.isdir(directory):
        os.mkdir(directory)
        
    sc.misc.imsave( directory  + fileName + '.png', img)                    


  
def getfIndex(nSamps, fs):
    
    f = np.fft.fftfreq(nSamps,1./fs)
#    nSamps=np.double(nSamps)
#    fs=np.double(fs)
#    nyq = fs/2
#    df = fs / nSamps
#    f = np.arange(nSamps) * df
#    f[f>nyq] = f[f>nyq] - nyq*2
    return f

def get2dCfIndex(xsamps,ysamps,fs):
    fx, fy = np.meshgrid( getfIndex(xsamps,fs), getfIndex(ysamps,fs) )
    C = fx + 1j * fy
    return C

def fft2Interpolate(coef, points, w):
    
    basis = np.exp( 1j * 2 * pi * ( points[0,0] * w[0] + points[0,0] * w[1] ))
    nPoints = np.size(points)/2
    intrpvals = np.zeros( nPoints, 'complex')
    for ind in xrange(nPoints):
        basis[:,:] = np.exp( 1j * 2 * pi * ( points[0, ind] * w[0] + points[1, ind] * w[1] ) )
        intrpvals[ind] = np.sum(  coef * basis  )
    return intrpvals

   
def translateByPixels(img,x,y):
    x = int(np.round(x))
    y = int(np.round(y))
    newImg= np.zeros(np.shape(img))
    nrows= np.size(img,0)
    ncols= np.size(img,1)
    r , c = np.meshgrid( range(nrows), range(ncols) );
    
    newrow = r-y
    newcol = c+x
    
    valid = (newrow<nrows) & (newcol<ncols) & (newcol>=0) & (newrow>=0)
    r =  r[valid]
    c =  c[valid]
    newrow = newrow[valid]
    newcol = newcol[valid]
    
    newImg[newrow,newcol] = img[r,c]
    
    return newImg

#def FT
  
def FTcutToNPixels(dR,dC,mat):
    nRows = np.size( mat, 0)
    nCols = np.size( mat, 1 )

    rCutT=np.ceil(dR/2)+1
    rCutB=nRows-np.floor(dR/2)+1;

    cCutL=np.ceil(dC/2)+1#the left column cut off
    cCutR=nCols-np.floor(dC/2)+1;#the right column cut off

    #take the pieces and put them into a smaller image
    top = np.concatenate((mat[:rCutT, :cCutL ], mat[ :rCutT, cCutR: ]),1)
    bottom = np.concatenate((mat[ rCutB:, :cCutL], mat[rCutB:,cCutR:]),1 )
    mat = np.concatenate((top,bottom),0)    
    
    return mat

def centeredPad( img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)
   
   #if an odd number defaults to putting assym pixel 
   
   hDif = ( new_width - width)/2.
   vDif = ( new_height - height   )/2.
   
   left = np.ceil(hDif)
   top = np.ceil(vDif)
   right = np.floor(hDif)
   bottom = np.floor(vDif)
   
   
   pImg = np.pad(img, ( (left, right), (top, bottom) ) ,'constant')
   return pImg
    
    
    
def centeredCrop(img, new_height, new_width):

   width =  np.size(img,1)
   height =  np.size(img,0)
   
   #if an odd number defaults to putting extra pixel to left and top 
   left = np.ceil((width - new_width)/2.)
   top = np.ceil((height - new_height)/2.)
   right = np.floor((width + new_width)/2.)
   bottom = np.floor((height + new_height)/2.)
   
   
   cImg = img[top:bottom, left:right]
   return cImg
    

def guassianDownSampleSTD( oldSize, newSize,  stdCutOff, fs ):
    #choose your low pass filter for downsampling based off some STD
    #if you choose 1 std 33% of energy chopped of when downsampling
    #if you choose 3 0.1 %
    oldSize, newSize, stdCutOff, fs = np.double([oldSize, newSize, stdCutOff, fs])
    
    return ((fs)*(newSize/oldSize))/stdCutOff
    
#def fftDilateImg
    
def fftResampleImg(img, nPix, stdCutOff = 4):
    #this is only for square images
    oldSize = np.size(img,0)

#    sr = sig.resample(img, nPix)
#    sr = sig.resample(sr, nPix, axis = 1)  
    std = guassianDownSampleSTD( oldSize, nPix,  stdCutOff, oldSize )
    sr = sig.resample(img, nPix, window = ('gaussian',std))
    sr = sig.resample(sr, nPix, window = ('gaussian',std), axis = 1)

        
    return sr

def fftDilateImg(img, dilR ):
    #just for square images for now
    nPix= np.array(np.shape(img))
    n = np.round(nPix*dilR)
    efRatioX= n[1]/np.double(nPix[1]) 
    efRatioY= n[1]/np.double(nPix[1]) 
    
    if np.double(n[1]/nPix[0]) != n[0]/ np.double(nPix[0]):
        warnings.warn( 'There will be a small distortion, percent '+ str(100*(efRatioY/efRatioX) ))
    
    temp = fftResampleImg(img, n[0], stdCutOff = 4)
    if (np.size(temp)>np.size(img)):  
    
        dilImg = centeredCrop(temp, nPix[0], nPix[1])
    
    else:
        dilImg = centeredPad(temp, nPix[0], nPix[1])    
    
    return dilImg
    

def imgStackTransform(imgDict, trans_stack):
    
    for ind in range(np.size(trans_stack,0)):
        trans_img = trans_stack[ind,:,:]
        
        if 'scale' in imgDict:
            trans_img = fftDilateImg(trans_img, imgDict['scale'][ind] )
        
        if 'rot' in imgDict:
            trans_img = scipy.misc.imrotate(trans_img, imgDict['rot'][ind], interp='bilinear')
   
        if 'x' and 'y' in imgDict:
            x = imgDict['x'][ind] 
            y = imgDict['y'][ind] 
            trans_img = translateByPixels(trans_img, x, y)
            
        elif 'x'  in imgDict:
            x = imgDict['x'][ind] 
            trans_img = translateByPixels(trans_img, x, np.zeros(np.shape(x)))
            
        elif 'y'  in imgDict:
            y = imgDict['y'][ind]
            trans_img = translateByPixels(trans_img, x, np.zeros(np.shape(x)))
            
        trans_stack[ind,:,:] = trans_img
    return trans_stack

##check the dilation function
#import matplotlib.pyplot as plt
#plt.close('all')
#
#img = np.load('/Users/dean/Desktop/AlexNet_APC_Analysis/stimSubset/2000.npy')
#img = img[0,:,:]
#rImg = fftResampleImg(img, 222, 4 )
#
#rImg = rImg - np.min(rImg)
#srImg = (rImg/np.max(rImg))*255
#
#plt.subplot(311)
#plt.imshow(img, interpolation = 'none', cmap = plt.cm.Greys_r )
#plt.title('Original Image')
#
#plt.subplot(312)
#plt.imshow(srImg, interpolation = 'none', cmap = plt.cm.Greys_r  )
#plt.title('Downsampled Image')
#
#cImg = fftDilateImg(srImg, 0.2 )
#plt.subplot(313)
#plt.imshow(cImg, interpolation = 'none', cmap = plt.cm.Greys_r  )
#plt.title('Shrunk Downsampled Image')
