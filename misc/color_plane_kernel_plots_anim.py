# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:52:06 2016

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:41:52 2016

@author: deanpospisil
"""

import pickle
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Cellar/ffmpeg/2.8.4/bin'



if 'a' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            a = pickle.load(f, encoding='latin1')
        except:
            a = pickle.load(f)

def frac_var_chrom(im):
    tot_var = np.linalg.norm(im)**2
    white_comp = np.tile(np.mean(im, 2, keepdims=True), (1,1,3))
    color_comp = im - white_comp
    color_var = np.linalg.norm(color_comp)**2    
    return color_var/tot_var
    

plt.close('all')
# convert these to circular
#for cind in [55,56,82,95,60,59,94]:

layer = 0    
ims = a[layer][1]
ims = np.array([im for im in ims])
low_rank = []
q,r = np.linalg.qr(np.array([[1,1,1],[1,1,0],[0,1,1]]).T)

fig = plt.figure()
ax = plt.gca()


def animate(cind):
    im = ims[cind,:,:]
    im = np.swapaxes(im, 0, 2)
    im = im.astype(np.float64)
    im=im/np.linalg.norm(im)
    imn = im - np.min(im)
    imn = imn / np.max(imn)
    
    plt.subplot(211)
    plt.imshow(imn, interpolation='nearest')
    
    rgb = np.reshape(im, (11*11, 3))
    chrom_coords = np.dot(rgb, q[:,1:])
    rgb = rgb - np.min(rgb)
    rgb = rgb / np.max(rgb)
    ax.figure.canvas.draw()
    plt.subplot(212)

    plt.cla()
    plt.axis('square')
    plt.xlim(-0.25, 0.25)
    plt.ylim(-0.25, 0.25)
    plt.scatter(x=chrom_coords[:,0].T, y=chrom_coords[:,1], s=1, 
            color=rgb)


im_ani = FuncAnimation(fig, animate, interval=500, frames=95, blit=False)

FFwriter = animation.FFMpegWriter()
im_ani.save('basic_animation.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
