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

adjust_c = 4 # cuvature values weren't matching files I got so I scaled them
downsamp = 5
shape_dict_list = [{'curvature': -((2. / (1 + np.exp(-0.125*dc.curve_curvature(cs)*adjust_c)))-1)[::downsamp],
                    'orientation': ((np.angle(dc.curveAngularPos(cs)))%(np.pi*2))[::downsamp]}
                    for cs in
                    map(lambda shape: shape[:, 1]*1j + shape[:, 0], s)]

import apc_model_fit as ac
maxAngSD = np.deg2rad(171)
minAngSD = np.deg2rad(23)
maxCurSD = 0.98
minCurSD = 0.09
nMeans = 16
nSD = 16
fn = 'apc_models_mycurve.nc'
'''
dmod = ac.make_apc_models(shape_dict_list, range(370), fn, nMeans, nSD, 
                          maxAngSD, minAngSD, maxCurSD, minCurSD, 
                          prov_commit=False)['resp']
'''
import pickle
ind = -1
def key_event(e):

    global ind
    ind = ind+1
    print(ind)
    plt.gca().cla()
    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation']), shape_dict_list[ind]['curvature'])
    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation'][0]), shape_dict_list[ind]['curvature'][0], color='g')
    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation'][20]), shape_dict_list[ind]['curvature'][20], color='y')
    with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
        shape_dict_list2 = pickle.load(f)
    plt.scatter( np.rad2deg(shape_dict_list2[ind]['orientation']), shape_dict_list2[ind]['curvature'], color='r')
    plt.show()

#
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)

plt.show()
#
#ind = 155
#for ind in range(370):
#
#    plt.close('all')
#    plt.subplot(3,1,1)
#    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation']), shape_dict_list[ind]['curvature'])
#    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation'][0]), shape_dict_list[ind]['curvature'][0], color='g')
#    plt.scatter( np.rad2deg(shape_dict_list[ind]['orientation'][20]), shape_dict_list[ind]['curvature'][20], color='y')
#    import pickle
#    #open those responses, and build apc models for their shapes
#    with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
#        shape_dict_list2 = pickle.load(f)
#
#    plt.scatter( np.rad2deg(shape_dict_list2[ind]['orientation']), shape_dict_list2[ind]['curvature'], color='r')
#
#    plt.subplot(3,1,2)
#    perc_num = int(np.shape(s[ind])[0]/1.5)
#    plt.scatter(s[ind][1:perc_num,0],s[ind][1:perc_num,1])
#    plt.scatter(s[ind][0,0],s[ind][0,1], color='g')
#    x, y = dc.get_center_boundary(s[ind][:,0], s[ind][:,1])
#    plt.scatter(x, y, color='b')
#    plt.axis('equal')
#    plt.subplot(3,1,3)
#    plt.imshow(np.load(top_dir + 'net_code/images/baseimgs/PC370/' + str(int(ind)) +'.npy' ))
#
#    plt.show()

#comb=(dc.curve_curvature(cs))*dc.curveAngularPos(cs)
#comb = comb
#
#plt.quiver(np.real(cs),np.imag(cs), np.real(comb),np.imag(comb),width=0.001,)
#plt.scatter(np.real(cs),np.imag(cs))
#plt.axis('equal')
#


#
#bg.save_boundaries_as_image(s, saveDir + baseImage + '/', top_dir, max_ext,
#                         n_pix_per_side=227, fill=True, require_provenance=False,
#                         frac_of_image=frac_of_image, use_round=False)
