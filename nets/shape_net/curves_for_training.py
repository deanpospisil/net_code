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
import d_misc as dm
import base_shape_gen as bg
import scipy.stats as st
from scipy import ndimage
from collections import OrderedDict as ord_d

#takes a set of points in apc plane and makes prediction based on different receptive fields
def apc_models(shape_dict={'curvature': None, 'orientation': None},
                                model_params_dict={'or_sd': [3.14],
                                                   'or_mean':[3.14],
                                                   'cur_mean':[1],
                                                   'cur_sd':[0.1]}):

    #initialize our distributions
    von_rv = st.vonmises( kappa=np.expand_dims(model_params_dict['or_sd']**-1, 1) ,
                         loc = np.expand_dims(model_params_dict['or_mean'],1 ))
    norm_rv = st.norm( scale = np.expand_dims(model_params_dict['cur_sd'],1) ,
                      loc = np.expand_dims(model_params_dict['cur_mean'], 1))

    model_resp_all_apc_points = (von_rv.pdf(np.expand_dims(shape_dict['orientation'], 0)) *
                                           norm_rv.pdf(np.expand_dims(shape_dict['curvature'],0)))
    model_resp = np.max(model_resp_all_apc_points, axis=1)

    return model_resp


def boundary_transform(transform_dict, boundary_set, apc_set):
    shape_id = int(transform_dict['shapes'])
    the_cboundary = cboundary_set[shape_id]
    curv = apc_set[int(transform_dict['shapes'])]['curvature']
    ori = apc_set[int(transform_dict['shapes'])]['orientation']

    if 'rotation' in transform_dict:
        the_cboundary = the_cboundary * np.exp(1j * transform_dict['rotation'])
        ori = (transform_dict['rotation'] + ori)%(2*np.pi)

    if 'scale' in transform_dict:
        the_cboundary= the_cboundary * transform_dict['scale']

    if 'x'  in transform_dict:
        the_cboundary = transform_dict['x'] + the_cboundary

    if 'y'  in transform_dict:
        the_cboundary = transform_dict['y']*1j + the_cboundary

    transformed_boundary = np.hstack((np.real(the_cboundary), np.imag(the_cboundary)))

    return transformed_boundary, ori, curv


def boundary_to_mat_by_round(s, n_pix_per_side, fill=True):
    im = np.zeros((n_pix_per_side, n_pix_per_side))
    #tr = scale_center_boundary_for_mat(s, n_pix_per_side, frac_of_image, max_ext)
    tr = s.astype(int)

    #conversion of x, y to row, col
    im[(n_pix_per_side-1)-tr[:, 1], tr[:, 0]] = 1

    if fill:
        im = ndimage.binary_fill_holes(im).astype(int)

#        if not im[tuple(np.median(tr,0))] == 1:
#            raise ValueError('shape not bounded')
    return im

dm.ifNoDirMakeDir(top_dir + 'net_code/data/train_img/')
unique_over_rot_scale = [1, 2, 10, 16, 18, 30, 34, 42, 50, 58, 66, 74, 82, 90, 98,
                    106, 114, 122, 130, 138,  146, 154, 162, 170, 178, 186, 194,
                    202, 210, 218, 226, 228, 236, 240, 248, 256, 258, 266, 274,
                    282, 290, 298, 306, 314, 322, 330, 338, 346, 354, 362]
mat = l.loadmat(top_dir + 'net_code/img_gen/'+ 'PC3702001ShapeVerts.mat')
boundary_set = bg.center_boundary([the_shape[:-1,:] for the_shape in np.array(mat['shapes'][0])])


with open(top_dir + 'net_code/data/models/PC370_params.p', 'rb') as f:
    apc_set = pickle.load(f)



#now create the transformations
frac_img = 0.5
n_pix_per_side = 64
max_ext = np.max(np.abs(np.vstack(boundary_set)))*2
center_im = round(n_pix_per_side/2.)
max_scale = (frac_img*n_pix_per_side)/max_ext
min_pos = (max_ext * max_scale)/2. + 1
max_pos = n_pix_per_side - min_pos - 1
cboundary_set= [np.expand_dims(np.sum(a_shape * [1, 1j], 1), 1) for a_shape in boundary_set]

'''
stim_trans_cart_dict, _ = cf.stim_trans_generator(shapes=unique_over_rot_scale,
                                                  scale=(max_scale/1.5, max_scale, 2),
                                                  x=(min_pos, max_pos, 16),
                                                  y=(min_pos, max_pos, 16),
                                                  rotation=(0, 2*np.pi-(2*np.pi/16), 16))



#make the models to be fit


model_params_dict = ord_d({'or_sd': np.linspace(np.deg2rad(27), np.deg2rad(180), 5),
                    'or_mean':np.linse(np.deg2rad(0), np.deg2rad(360-360/5.), 5),
                    'cur_mean':np.linspace(-0.5, 1, 5),
                    'cur_sd':np.linspace(0.1, 1, 5)})
cart_params_dict = dm.cartesian_prod_dicts_lists( model_params_dict )
'''
stim_trans_cart_dict, _ = cf.stim_trans_generator(shapes=unique_over_rot_scale,
                                                  scale=(max_scale/1.5, max_scale, 1),
                                                  x=(min_pos, max_pos, 1),
                                                  y=(min_pos, max_pos, 1),
                                                  rotation=(0, 2*np.pi-(2*np.pi/16), 16))

cart_params_dict = ord_d({'or_sd':np.array([0.2]),
                    'or_mean':np.array([0]),
                    'cur_mean':np.array([1]),
                    'cur_sd':np.array([0.2])})
m=[]
im=[]
shape_dict_list_trans = {}
shape_dict_list_trans['orientation'] = []
shape_dict_list_trans['curvature'] = []
for ind in range(len(stim_trans_cart_dict['shapes'])):
    print(np.double(ind)/len(stim_trans_cart_dict['shapes']))
    transform_dict = {key: stim_trans_cart_dict[key][ind] for key in stim_trans_cart_dict.keys() }
    ts, ori, curv = boundary_transform(transform_dict, cboundary_set, apc_set)
    shape_dict_list_trans['curvature'] = curv
    shape_dict_list_trans['orientation'] = ori

    #choose some models
    m.append(apc_models(shape_dict=shape_dict_list_trans,
                        model_params_dict=cart_params_dict))
    #get the image
    _ =  boundary_to_mat_by_round(ts, n_pix_per_side=n_pix_per_side, fill=True)
    _ = _-np.mean(_)
    _ = _ / np.linalg.norm(_)
    im.append(_)
    



data_dir = 'data/train_img/'

#data = np.zeros((n_imgs, 1, img_width, img_width))
data = np.expand_dims(np.array(im), 1)
data = data.astype('float32')

#targets = np.ones((n_imgs, nunits))
targets = np.array(m)
targets = targets.astype('float32')


import h5py
with h5py.File(top_dir + 'net_code/' + data_dir + 'train_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = targets

with open(top_dir + 'net_code/nets/shape_net/train_data_list.txt', 'w') as f:
    f.write(top_dir + 'net_code/' + data_dir + 'train_data.h5')


with h5py.File(top_dir+ 'net_code/' + data_dir + 'test_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = targets

with open(top_dir + 'net_code/nets/shape_net/test_data_list.txt', 'w') as f:
    f.write(top_dir+ 'net_code/' + data_dir + 'test_data.h5' )



