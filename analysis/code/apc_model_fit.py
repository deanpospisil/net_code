# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:28:57 2016

@author: dean
"""

# analysis
import scipy.stats as st
import numpy as np
import warnings
import os
import sys
from collections import OrderedDict as ord_d
import matplotlib.pyplot as plt

top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)
import xarray as xr
import d_misc as dm
import pickle

def get_2d_dims_right(vec, dims_order=(1,0)):
    dims = vec.shape
    if len(dims)>2:
        warnings.warn('model params should not have more than 2-d')
        right_dims = None

    elif len(dims) < 2 :
        right_dims = np.expand_dims(vec, axis=dims_order[1])

    elif dims[ dims_order[1]] < dims[ dims_order[0]]:
        right_dims = np.swapaxes( vec, 1, 0)

    return right_dims

#takes a set of points in apc plane and makes prediction based on different receptive fields
def apc_models( shape_dict_list = [{'curvature': None, 'orientation': None} ],
                                   model_params_dict = { 'or_sd': [3.14], 'or_mean':[3.14], 'cur_mean':[1], 'cur_sd':[0.1]} ):

     # make sure everything has the right dimensionality for broadcating
    for key in model_params_dict:
        vec = np.array( model_params_dict[key] )
        model_params_dict[key] = get_2d_dims_right(vec, dims_order=(1,0) )


    # make sure everything has the right dimensionality for broadcating, figure out a more succint way to do this
    for ind, a_shape in enumerate(shape_dict_list):
        for key in a_shape:
            vec = np.array(a_shape[key])
            a_shape[key] = get_2d_dims_right(vec, dims_order= (0,1) )

        shape_dict_list[ind] = a_shape

    #initialize our distributions
    von_rv = st.vonmises( kappa = model_params_dict['or_sd']**-1 , loc = model_params_dict['or_mean'] )
    norm_rv = st.norm( scale = model_params_dict['cur_sd'] , loc = model_params_dict['cur_mean'] )

    #get responses to all points for each axis ap and c then their product, then the max of all those points as the resp
    model_resp_all_apc_points = [ von_rv.pdf( apc_points['orientation'] ) * norm_rv.pdf( apc_points['curvature'] ) for apc_points in shape_dict_list ]
    model_resp = np.array([ np.max( a_shape, axis = 0 ) for a_shape in model_resp_all_apc_points])

    #mean subtract
    model_resp = model_resp - np.mean( model_resp, axis = 0 )
    #scale
    magnitude = np.linalg.norm( model_resp, axis = 0)
    model_resp = model_resp / magnitude

    return model_resp
    
def make_apc_models(shape_dict_list, fn, nMeans, nSD, maxAngSD, minAngSD, maxCurSD, minCurSD,
                    prov_commit = False):
    #make this into a pyramid based on d-prime
    orMeans = np.linspace(0, 2*np.pi - 2*np.pi / nMeans, nMeans)
    orSDs = np.logspace(np.log10( minAngSD ), np.log10( maxAngSD ), nSD )
    curvMeans = np.linspace( -0.5, 1, nMeans )
    curvSDs = np.logspace( np.log10(minCurSD), np.log10(maxCurSD), nSD )
    
    
    model_params_dict = ord_d({'or_sd': orSDs, 'or_mean':orMeans,
                         'cur_mean' :curvMeans, 'cur_sd':curvSDs})
    
    model_params_dict = dm.cartesian_prod_dicts_lists( model_params_dict )
    
    
    model_resp = apc_models(shape_dict_list=shape_dict_list, 
                            model_params_dict=model_params_dict)
    
    dam = xr.DataArray(model_resp, dims = ['shapes', 'models'])
    
    for key in model_params_dict.keys():
        dam[key] = ('models', np.squeeze(model_params_dict[key]))
    
    if prov_commit:
        sha = dm.provenance_commit(top_dir)
        dam.attrs['model'] = sha
    
    
    ds = xr.Dataset({'resp': dam})
    ds.to_netcdf(top_dir + 'analysis/data/models/' + fn)
    return dam

with open(top_dir + 'images/baseimgs/PC370/PC370_params.p', 'rb') as f:
    shape_dict_list = pickle.load(f)

maxAngSD = np.deg2rad(171)
minAngSD = np.deg2rad(23)
maxCurSD = 0.98
minCurSD = 0.09

nMeans = 3
nSD = 3
fn = 'apc_models_test.nc'
dam = make_apc_models(shape_dict_list, fn, nMeans, nSD, maxAngSD, minAngSD, maxCurSD, minCurSD,
                prov_commit=True)
                
