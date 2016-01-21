# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:04:21 2015

@author: dean
"""
import sys
import numpy as np
import scipy.io as  l
import scipy
import scipy as sc
import matplotlib.pyplot as plt
import os
import pickle
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)
import d_curve as dc
import d_misc as dm


def boundaryToMat(boundary, nPixPerSide = 227, fill = True ):
        
    #haven't quite figured out how to get the correct images.
    plt.close('all')
    inchOverPix = 2.84/227. #this happens to work because of the dpi of my current screen. 1920X1080
    inches = inchOverPix*nPixPerSide
    
    if inches<0.81:
        print( 'inches < 0.81, needed to resize')
        tooSmall = True
        inches = 0.85
    
    fig=plt.figure(figsize = ( inches, inches ))#min size seems to be 0.81 in the horizontal, annoying
    
    plt.axis( 'off' )
    plt.gca().set_xlim([-1, 1])
    plt.gca().set_ylim([-1, 1])
    
    if fill is True:
        line = plt.Polygon(boundary, closed=True, fill='k', edgecolor='none',fc='k')
    else:
        line = plt.Polygon(boundary, closed=True, fill='k', edgecolor='k',fc='w')
        
    plt.gca().add_patch(line)

    fig.canvas.draw()
    
    data1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data2 = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    
    data2[data2 == data2[0,0,0]] = 255
    ima = - (data2 - 255)[:,:,0]
    
    if np.size( ima, 0 ) is not nPixPerSide and np.size(ima,1 ) is not nPixPerSide:
        print('had to resize')
        ima = scipy.misc.imresize(ima, (nPixPerSide, nPixPerSide), interp='cubic', mode=None)


    return ima


def save_boundaries_as_image( imlist, save_dir,cwd, nPixPerSide = 227 ,  fill = True, require_provenance = False ):

    dir_filenames = os.listdir(save_dir)
    
    #remove existing files
    for name in dir_filenames:
        if 'npy' in name or 'png' in name or 'pickle' in name:
            os.remove(save_dir + name)
    
    
    if require_provenance is True:
        
        #commit the state of the directory and get is sha identification
        sha = dm.provenance_commit(cwd)
    
        #now save that identification with the images
        sha_file = save_dir + 'sha1'
        with open( sha_file + '.pickle', 'wb') as f:
            pickle.dump( sha, f )

    for boundaryNumber in range(len(imlist)):
    
        im = boundaryToMat(imlist[boundaryNumber], nPixPerSide, fill  )
        sc.misc.imsave( save_dir + str(boundaryNumber) + '.bmp', im)
        np.save( save_dir  + str(boundaryNumber) , im)
        
    
def centerBoundary(s):
    
    #centroid, center of mass, https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    for ind in range(len(s)):
        minusone = np.arange(-1, np.size(s[ind],0)-1)
        y = s[ind][:,1]
        x = s[ind][:,0]
        A = 0.5*np.sum( x[minusone]*y[:] - x[:]*y[minusone])
        normalize= (1/(A*6.))
        cx = normalize * np.sum( (x[minusone] + x[:] ) * (x[minusone]*y[:] - x[:]*y[minusone]) )
        cy = normalize * np.sum( (y[minusone] + y[:] ) * (x[minusone]*y[:] - x[:]*y[minusone]) )
        s[ind][:,1] = y - cy
        s[ind][:,0] = x - cx
    
    return s
def scaleBoundary(s, fracOfImage):
    
    
    #get the furthest point from the center
    ind= -1
    curmax = 0  
    for sh in s:
        ind+=1
        testmax = np.max(np.sqrt( np.sum(sh**2, 1) ))
        if curmax<testmax:
            curmax = testmax
    
    #or maybe the furthest x y
   
    #scale the shape so the furthest point from the center is some fraction of 1
    scaling=curmax/fracOfImage     
    for ind in range(len(s)):
        s[ind] = s[ind]/scaling
        
    return s



#generate base images 

saveDir = cwd + '/images/baseimgs/'

dm.ifNoDirMakeDir(saveDir)

baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseImage = baseImageList[0] 
fracOfImage = 1.20
dm.ifNoDirMakeDir(saveDir + baseImage +'/')


if baseImage is baseImageList[0]:

#    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(cwd + '/imggen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])

elif baseImage is baseImageList[1]:

    s = dc.make_n_natural_formlets( n=10,
                nPts=600, radius=1, nFormlets=32, meanFormDir=np.pi/2, 
                stdFormDir=np.pi/3, meanFormDist=1, stdFormDist=0.1, 
                startSigma=3, endSigma=0.1, randseed = 2 )

elif baseImage is baseImageList[2]:
    print('to do')
    
elif baseImage is baseImageList[3]:
    print('to do')
    
    
s = centerBoundary( s )
s = scaleBoundary ( s, fracOfImage )
save_boundaries_as_image( s, saveDir + baseImage + '/', cwd, nPixPerSide = 540 ,  fill = True, require_provenance = True )