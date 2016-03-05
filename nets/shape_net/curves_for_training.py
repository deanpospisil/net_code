# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:51:58 2016

@author: dean
"""
import sys
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append(top_dir + 'net_code/img_gen')
sys.path.append( top_dir + 'xarray/')

import d_curve as dc
import d_misc as dm
import d_img_process as imp
import base_shape_gen as bg

saveDir = top_dir + 'net_code/images/baseimgs/'
dm.ifNoDirMakeDir(saveDir)

baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseImage = baseImageList[0]

frac_of_image = 0.25
dm.ifNoDirMakeDir(saveDir + baseImage +'/')

if baseImage is baseImageList[0]:

#    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(top_dir + 'net_code/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    s = [shape[:-1,:] for shape in s]

elif baseImage is baseImageList[1]:
    nPts = 1000
    s = dc.make_n_natural_formlets(n=10,
                nPts=nPts, radius=1, nFormlets=32, meanFormDir=np.pi,
                stdFormDir=2*np.pi, meanFormDist=1, stdFormDist=0.1,
                startSigma=3, endSigma=0.1, randseed=1, min_n_pix=64,
                frac_image=frac_of_image)
elif baseImage is baseImageList[2]:
    #    os.chdir( saveDir + baseImageList[0])
    mat = l.loadmat(top_dir + 'net_code' + '/img_gen/'+ 'PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    #adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
    a = np.hstack((range(14), range(18,318)))
    a = np.hstack((a, range(322, 370)))
    s = s[a]
    s = [shape[:-1,:] for shape in s]

elif baseImage is baseImageList[3]:
    print('to do')

s = bg.center_boundary(s)
shape_dict_list = [{'curvature': dc.curve_curvature(cs),
                    'orientation': dc.curveAngularPos(cs)}
                    for cs in
                    map(lambda shape: shape[:, 1] + shape[:, 0]*1j, s)]

#
#comb=(magnitude)*direction
#comb = comb
#cshape = cs
#plt.quiver(np.real(cshape),np.imag(cshape), np.real(comb),np.imag(comb),width=0.001,)
#plt.scatter(np.real(cshape),np.imag(cshape))
#plt.axis('equal')

#max_ext = np.max([np.max(np.abs(a_s)) for a_s in s])



#
#bg.save_boundaries_as_image(s, saveDir + baseImage + '/', top_dir, max_ext,
#                         n_pix_per_side=227, fill=True, require_provenance=False,
#                         frac_of_image=frac_of_image, use_round=False)
