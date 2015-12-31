# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:40:24 2015

@author: deanpospisil
"""

import numpy as np
from collections import OrderedDict as ordDict

import os
import sys
import warnings

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)

import dImgProcess as imp

def  net_imgstack_response(net, stack):
    #stack is expected to be nImages x RGB x rows x cols

    if len( stack.shape ) < 3: #in case the stack is greyscale
        stack = np.tile(stack, (3,1,1,1))
        stack = np.swapaxes(stack, 0, 1)
        
    
    stackHeight = np.size( stack, 0 )
    layerNameList = [(k) for k, v in net.params.items()]
    
    #run the full stack through the net 
    net.blobs['data'].reshape( stackHeight, 3, 227, 227 )
    net.blobs['data'].data[...] = stack
    net.forward()
    
    #go through each layer getting unit respons
    for layerInd, layerName in zip( range(len(layerNameList)), layerNameList ):
        
        if layerName[0] == 'c':#if it is a convolutional layer
            mid = round(np.shape(net.blobs[layerName].data)[3] / 2)
            #get a list of responses from all unique units
        else:
            print('la')
    
    return response
            
            

def identity_preserving_transform_resp( img_stack, stim_specs_dict, net, nimgs_per_pass = 100 ):
    #takes stim specs, transforms images accordingly, gets their responses 
    
    n_imgs = len( stim_specs_dict[stim_specs_dict.keys[0]] )
    stack_indices, remainder = misc.sectStrideInds( nimgs_per_pass, n_imgs )
    
    #now divide the dict up into sects.
    #order doesn't matter using normal dict
    stim_specs_dict_sect[key] = {} 
    for stack_ind in stack_indices:
        
        #load up a chunk of images
        for key in stim_specs_dict:
            stim_specs_dict_sect[key] = stim_specs_dict[key][tuple(stack_ind)]
        
        trans_stack = imp.imgStackTransform( stim_specs_dict_sect, img_stack )
        
        net_resp = net_imgstack_response( net, trans_stack )
    
    return net_resp

def stim_idprestrans_generator(shape = None, scale = None, x = None, y = None, 
                        rotation = None):
    #either makes a ragged dict lists using nans, or makes a full cartesian 
    #product dictionary
    stim_specs_dict = ordDict()                 
        
    if not shape is None:
        stim_specs_dict[ 'shape' ] = map(float, shape)
    
    if not scale is None :
        stim_specs_dict[ 'scale' ] = np.linspace( *scale )

    if not x is None :
        stim_specs_dict[ 'x' ] = np.linspace( *x )
    
    if not y is None :
        stim_specs_dict[ 'y' ] = np.linspace( *y )
    
    if not rotation is None :
        stim_specs_dict[ 'rotation' ] = np.linspace( *rotation )

    stim_trans_dict = cartesian_prod_dicts_lists( stim_specs_dict )
    
    if not shape is None:
        stim_specs_dict[ 'shape' ] = map(int, stim_specs_dict[ 'shape' ])
 
    return stim_trans_dict
         
def cartesian_prod_dicts_lists( the_dict ) :
    
    if not type(the_dict) is type(ordDict()):
        warnings.warn('We were expecting an ordered dict for provenance concerns.')
        
    from sklearn.utils.extmath import cartesian
    
    stim_list = []
    stim_list = tuple([ list(the_dict[ key_name ]) for key_name in the_dict ])
        
    
    stim_cart_array = cartesian(stim_list)
    
    cart_dict = {}
    for key_name, key_num in zip( the_dict, range( len( the_dict ) ) ):
        cart_dict[key_name] = stim_cart_array[ :, key_num]
    
    
    
    return cart_dict
    
#d = ordDict()
#d['a'] = [1,2,3]
#d['b'] = [1,2]
#c_d = cartesian_prod_dicts_lists( d )

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)
    #
    
def load_sorted_dir_numbered_fnms_with_particular_extension(the_dir, extension):
      
    dir_filenames = os.listdir(the_dir)
    file_names = [ file_name for file_name in dir_filenames if file_name.split('.')[-1] == extension ]
    
    file_names = sorted( file_names, key = lambda num : int(num.split('.')[0]) )
    
    return file_names

def load_npy_img_dirs_into_stack( img_dir ):
    
    stack_descriptor_dict = {}
    img_names = load_sorted_dir_numbered_fnms_with_particular_extension( img_dir , 'npy')
    
    #will need to check this for color images.
    stack = np.array([ np.load( img_dir + img_name) for img_name in img_names ])
    stack_descriptor_dict['img_paths'] = [ img_dir + img_name for img_name in img_names ]
    
    #to do, some descriptor of the images for provenance: commit and input params for base shape gen
    #stack_descriptor_dict['base_shape_gen_inputs'] = [ img_dir + img_name for img_name in img_names ]

    return stack, stack_descriptor_dict
    
stim_trans_dict = stim_idprestrans_generator(shape = [1,2,3], scale = (0,1,4), x = (-1,1,4), y = None, rotation = None)
stack, stack_desc = load_npy_img_dirs_into_stack( '/Users/deanpospisil/Desktop/net_code/images/baseimgs/formlet/' )

ANNDir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ANNFileName='bvlc_reference_caffenet.caffemodel'
net_resp = identity_preserving_transform_resp( stack, stim_trans_dict, ANNDir+ANNFileName)