# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:12:25 2015

@author: dean
"""

import numpy as np
import scipy.io as  l
import scipy as sc
import matplotlib.pyplot as plt
    
mat = l.loadmat('PC3702001Verts.mat')
s = np.array(mat['shapes'][0])

#shape 1 will be our reference fo current params it is radius = 132 pixels
#desired pix size of circle
startdiamPix = 132.
diamPix = 150.
scale = startdiamPix/diamPix


#center everything
imlist=[]
for ind in range(len(s)):
    imlist.append((s[ind]-np.mean(s[ind],0))/scale)
    



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
    data1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data2 = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data2[data2 == data2[0,0,0]] = 255
    ima = - (data2 - 255)[:,:,0]
    toSave = ima
    sc.misc.imsave('PC370/' + str(ind) + '.png', toSave)
    np.save('PC370/' + str(ind) , toSave)

        
 


