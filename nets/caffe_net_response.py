# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:40:24 2015

@author: deanpospisil
"""

import numpy as np
from collections import OrderedDict as ordDict
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
                        rotation = None, cart = True):
    #either makes a ragged dict lists using nans, or makes a full cartesian 
    #product dictionary
    stim_specs_dict = ordDict()                 
        
        if not shape is None and  :
            stim_specs_dict[ 'shape' ] = shape
        
        if not scale is None :
            stim_specs_dict[ 'scale' ] = np.linspace( scale )
    
        if not x is None :
            stim_specs_dict[ 'x' ] = np.linspace( x )
        
        if not y is None :
            stim_specs_dict[ 'y' ] = np.linspace( y )
        
        if not rotation is None :
            stim_specs_dict[ 'rotation' ] = np.linspace( rotation )
        
        if cart is True: 
            
            stim_trans_dict = cartesian_prod_dicts_lists( stim_specs_dict )
 
        else:     
            print('not implemented yet')
            #stim_trans_dict = ragged_dicts_list()
            
    return stim_trans_dict
         
def cartesian_prod_dicts_lists( the_dict ) :
    
    from sklearn.utils.extmath import cartesian
    
    stim_vec_list = []

    
    for key_name, key_num in zip ( stim_specs_dict, range( len( stim_specs_dict ) ) ):
        stim_list[key_num] = stim_specs_dict[ keyname ]
    
    stim_cart_array = cartesian(stim_list)
    
    for key_name, key_num in zip ( stim_specs_dict, range( len( stim_specs_dict ) ) ):
        cart_dict[key_name] = stim_cart_array[ :, key_num]
    
    return cart_dict
    
    
        
                
                
        
    
    
    
