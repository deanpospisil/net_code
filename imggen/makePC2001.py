# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:55:03 2015

@author: dean
"""

from PIL import Image
import numpy as np
import scipy.io as  l
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
    
mat = l.loadmat('PC3702001Verts.mat')
s = np.array(mat['shapes'][0])

#shape 1 will be our reference fo current params it is radius = 132 pixels
#desired pix size of circle
startdiamPix = 132.
diamPix = 40.
scale = startdiamPix/diamPix

# positions
npts=21
theEnd = 75
#a = np.linspace( theEnd/npts, theEnd, npts ).reshape( npts, 1 ) 
#b = np.vstack( (a, -a ))
#b = np.sort(b,0)
#c = np.vstack( (b,  np.zeros( (np.size(b),1) ) ) )
#d = np.hstack((c,np.flipud(c)) )
#pos = np.vstack(( np.zeros( (1,2) ) , d ))
#a = b = c = d = None

a= np.linspace(-theEnd,theEnd,npts)
pos = np.vstack((a, np.zeros(np.shape(a)))).T



#center everything
imlist=[]
for ind in range(len(s)):
    imlist.append((s[ind]-np.mean(s[ind],0))/scale)

totInd=-1
for ind in range(len(imlist)):

    plt.close('all')
    fig=plt.figure(figsize = ( 2.84, 2.84 ))#this happens to give me 227x227 because of dpi of the screen
    points = imlist[ind]
    line = plt.Polygon(points, closed=True, fill='k', edgecolor='none',fc='k')
    plt.gca().add_patch(line)
    plt.axis('off')
    plt.gca().set_xlim([-2, 2])
    plt.gca().set_ylim([-2, 2])
    fig.canvas.draw()
    #plt.savefig('origStimuli/shape' + str(ind) + '.raw', dpi=100 )
    # Now we can save it to a numpy array.
    data1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data2 = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data2[data2 == data2[0,0,0]] = 255
    ima = - (data2 - 255)[:,:,0]

    #here I can do translations
    for tInd in range(np.size( pos, 0)):
        
        #print(totInd/ np.double(np.size(pos,0) * 370))
        
        totInd+=1
        print(totInd)
        trans = translateByPixels(ima, pos[tInd, 0], pos[tInd, 1])
        toSave = np.tile(trans,(3,1,1))
        
        #sc.misc.imsave('stimuli/' + str(totInd) + '.png', trans)
        sc.misc.imsave('stimSubset/' + str(totInd) + '.png', trans)
        np.save('stimSubset/' + str(totInd) , toSave)

        
 


