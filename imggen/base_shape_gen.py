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
import d_img_process as imp


def boundaryToMat(boundary, n_pix_per_side = 227, fill = True, frac_of_img=1 ):
    n_pix_per_side_old = n_pix_per_side
    if fracOfImage > 1:

        n_pix_per_side = round(n_pix_per_side*frac_of_img)

    plt.close('all')
    inchOverPix = 2.84/227. #this happens to work because of the dpi of my current screen. 1920X1080
    inches = inchOverPix*n_pix_per_side

    if inches<0.81:
        print( 'inches < 0.81, needed to resize')
        tooSmall = True
        inches = 0.85

    fig = plt.figure(figsize = ( inches, inches ))#min size seems to be 0.81 in the horizontal, annoying

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
    ima = imp.centeredCrop(ima, n_pix_per_side_old, n_pix_per_side_old)

    if (not np.size( ima, 0 ) == n_pix_per_side_old) or (not np.size(ima,1 ) == n_pix_per_side_old):
        print('had to resize')
        ima = scipy.misc.imresize(ima, (n_pix_per_side_old, n_pix_per_side_old), interp='cubic', mode=None)


    return ima


def save_boundaries_as_image( imlist, save_dir, cwd, n_pix_per_side = 227 ,  fill = True, require_provenance = False, fracOfImage=1 ):
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
        print(boundaryNumber)


        im = boundaryToMat(imlist[boundaryNumber], n_pix_per_side, fill, fracOfImage  )

        sc.misc.imsave( save_dir + str(boundaryNumber) + '.bmp', im)
        np.save( save_dir  + str(boundaryNumber) , im)

def get_center_boundary(x,y):
    minusone = np.arange(-1, np.size(x)-1)
    A = 0.5*np.sum( x[minusone]*y[:] - x[:]*y[minusone])
    normalize= (1/(A*6.))
    cx = normalize * np.sum( (x[minusone] + x[:] ) * (x[minusone]*y[:] - x[:]*y[minusone]) )
    cy = normalize * np.sum( (y[minusone] + y[:] ) * (x[minusone]*y[:] - x[:]*y[minusone]) )
    return cx, cy
def center_boundary(s):

    #centroid, center of mass, https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    for ind in range(len(s)):
        minusone = np.arange(-1, np.size(s[ind],0)-1)
        y = s[ind][:,1]
        x = s[ind][:,0]
        cx, cy = get_center_boundary(x,y)
        s[ind][:,0] = x - cx
        s[ind][:,1] = y - cy

    return s
def scaleBoundary(s, fracOfImage):

    if fracOfImage>1:
        fracOfImage =1

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
def flood_fill(image, x, y, value):
    "Flood fill on a region of non-BORDER_COLOR pixels."
    if not image.within(x, y) or image.get_pixel(x, y) == value:
        return
    edge = [(x, y)]
    image.set_pixel(x, y, value)
    while edge:
        newedge = []
        for (x, y) in edge:
            for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if image.within(s, t) and \
                	image.get_pixel(s, t) not in (value):
                    image.set_pixel(s, t, value)
                    newedge.append((s, t))
        edge = newedge


#generate base images

saveDir = cwd + '/images/baseimgs/'

dm.ifNoDirMakeDir(saveDir)

baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseImage = baseImageList[1]

pixCirc = 51
fracOfImage = (pixCirc/135.)


dm.ifNoDirMakeDir(saveDir + baseImage +'/')


if baseImage is baseImageList[0]:

#    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(cwd + '/imggen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])

elif baseImage is baseImageList[1]:

    s = dc.make_n_natural_formlets( n=10,
                nPts=2000, radius=1, nFormlets=32, meanFormDir=np.pi/2,
                stdFormDir=np.pi/3, meanFormDist=1, stdFormDist=0.1,
                startSigma=3, endSigma=0.1, randseed = np.random.randint(1000))
elif baseImage is baseImageList[2]:
    print('to do')

elif baseImage is baseImageList[3]:
    print('to do')


s = center_boundary(s)
s = scaleBoundary ( s, fracOfImage )
#save_boundaries_as_image( s, saveDir + baseImage + '/', cwd, n_pix_per_side = 227 ,  fill = True, require_provenance = True, fracOfImage = fracOfImage )
m = np.max(np.abs(s))
n_pix = 64.
frac_of_image = 0.5

scale = (n_pix*frac_of_image)/(m*2)
tr = np.round(np.array(s[5])*scale)

h = np.max(abs(tr[:,0]))
w = np.max(abs(tr[:,1]))

tr[:,0] = tr[:,0] + h + n_pix/4.
tr[:,1] = tr[:,1] + w + n_pix/4.

get_center_boundary()

tr = tr.astype(int)

im = np.zeros((n_pix,n_pix))

#im = np.zeros((5,5))
im[tr[:,1],tr[:,0]] = 1
from scipy import ndimage
im= ndimage.binary_fill_holes(im).astype(int)
plt.imshow(im, interpolation = 'none')



