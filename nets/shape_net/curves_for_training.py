# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:51:58 2016

@author: dean
"""
import sys, os
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'net_code/common')
sys.path.append(top_dir + 'net_code/img_gen')
sys.path.append(top_dir + 'net_code/nets')
sys.path.append( top_dir + 'xarray/')

import caffe_net_response as cf
import d_curve as dc
import d_misc as dm
import base_shape_gen as bg
import apc_model_fit as ac

frac_of_image = 0.25
dm.ifNoDirMakeDir(top_dir + 'net_code/train_img/')

mat = l.loadmat(top_dir + 'net_code' + '/img_gen/'+ 'PC3702001ShapeVerts.mat')
shapes = bg.center_boundary([the_shape[:-1,:] for the_shape in np.array(mat['shapes'][0])])

with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

#now create the transformations
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(370),
                                                             blur=None,
                                                             scale = (0.5, 1.5, 3),
                                                             x = (-50, 50, 16),
                                                             y = (-50, 50, 16),
                                                             rotation = (0, np.pi, 16))


def boundary_transform(transform_dict, boundary_set, apc_set):

    the_cboundary = np.sum(boundary_set[int(transform_dict['shapes'])] * [1, 1j], 1)
    the_apc_params = apc_set[int(transform_dict['shapes'])]

    if 'rotation' in transform_dict:
        the_cboundary = the_cboundary * np.exp(1j * transform_dict['rotation'])
        the_apc_params['orientation'] = (transform_dict['rotation'] +
                                        the_apc_params['orientation'])%(2*np.pi)
    if 'scale' in transform_dict:
        the_cboundary= the_cboundary * transform_dict['scale']



    if 'x'  in transform_dict:
        the_cboundary = transform_dict['x'] + the_cboundary

    if 'y'  in transform_dict:
        the_cboundary = transform_dict['y'] + 1j*the_cboundary

    transformed_boundary = np.array([np.real(the_cboundary), np.imag(the_cboundary)]).T
    return transformed_boundary

transform_dict = {key: stim_trans_cart_dict[key][1000000] for key in stim_trans_cart_dict.keys() }

transform_dict ={'shapes': 369, 'x': 0, 'rotation': 0, 'scale': 1.0, 'y': 0}
boundary_set = shapes
apc_set = shape_dict_list

ts = boundary_transform(transform_dict, boundary_set, apc_set)
plt.plot(ts[:,0], ts[:,1])
plt.axis('equal')

#shape_dict_list = [{'curvature': -((2. / (1 + np.exp(-0.125*dc.curve_curvature(cs)*adjust_c)))-1)[::downsamp],
#                    'orientation': ((np.angle(dc.curveAngularPos(cs)))%(np.pi*2))[::downsamp]}
#                    for cs in
#                    map(lambda shape: shape[:, 1]*1j + shape[:, 0], s)]

#
#maxAngSD = np.deg2rad(171)
#minAngSD = np.deg2rad(23)
#maxCurSD = 0.98
#minCurSD = 0.09
#nMeans = 16
#nSD = 16
#fn = 'apc_models_mycurve.nc'
#
#dmod = ac.make_apc_models(shape_dict_list, range(370), fn, nMeans, nSD,
#                          maxAngSD, minAngSD, maxCurSD, minCurSD,
#                          prov_commit=False)['resp']
#

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
'''