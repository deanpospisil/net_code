# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:07:05 2016

@author: deanpospisil
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys, os
top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'net_code/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

from moviepy.editor import VideoClip
import pickle 

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


def make_frame(cind):
    print(cind)
    
    im = ims[cind,:,:]
    im = np.swapaxes(im, 0, 2)
    im = im.astype(np.float64)
    im=im/np.linalg.norm(im)
    imn = im - np.min(im)
    imn = imn / np.max(imn)
    
    plt.subplot(211)
    plt.title('Kernel #: ' + str(int(cind)) + 
    ' chrom: ' + str(np.round(frac_var_chrom(im), decimals=3)) )
    plt.imshow(imn, interpolation='nearest')
    rgb = np.reshape(im, (11*11, 3))
    rgb = np.vstack((rgb, np.eye(3)*0.25))
    chrom_coords = np.dot(rgb, q[:,1:])
    rgb = rgb - np.min(rgb)
    rgb = rgb / np.max(rgb)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(212)

    plt.cla()
    plt.axis('square')
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, .3)
    plt.xticks([0,])
    plt.yticks([0,])
    plt.scatter(x=chrom_coords[:,0].T, y=chrom_coords[:,1], s=4, 
            color=rgb)

    ax.figure.canvas.draw()
    data1 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    frame_for_time_t = data1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return frame_for_time_t # (Height x Width x 3) Numpy array

animation = VideoClip(make_frame, duration=96) # 3-second clip

# For the export, many options/formats/optimizations are supported
animation.write_videofile("my_animation.mp4", fps=1) # export as video
#animation.write_gif("my_animation.gif", fps=1) # export as GIF (slow)