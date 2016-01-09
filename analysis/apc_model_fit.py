# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:28:57 2016

@author: dean
"""

# analysis
import scipy.io as  l
import scipy.stats as st
import numpy as np
import warnings
import os
import sys



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)

sys.path.append('/home/dean/caffe/python')
import xray as xr

import dMisc as dm

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
    vm_rv = st.vonmises( kappa = model_params_dict['or_sd']**-1 , loc = model_params_dict['or_mean'] ) 
    nm_rv = st.norm( scale = model_params_dict['cur_sd']**-1 , loc = model_params_dict['cur_mean'] ) 
    
    #get responses to all points for each axis ap and c then their product, then the max of all those points as the resp
    model_resp_all_apc_points = [ nm_rv.pdf( apc_points['orientation'] )*vm_rv.pdf( apc_points['curvature'] ) for apc_points in shape_dict_list ]
    model_resp = np.array([ np.max( a_shape, axis = 0 ) for a_shape in model_resp_all_apc_points])
    
    #mean subtract
    model_resp = model_resp - np.mean( model_resp, axis = 0 )
    #scale
    magnitude = np.linalg.norm( model_resp, axis = 0)
    magnitude = np.swapaxes(magnitude,0,1)
    model_resp = model_resp / magnitude

    return model_resp
    






mat = l.loadmat( '/Users/dean/Desktop/net_code/imggen/PC2001370Params.mat' )
s = mat['orcurv'][0]
shape_dict_list = []
for shape in s:
    shape_dict_list.append( { 'curvature' : shape[ : , 1 ], 'orientation' : shape[ : , 0 ]} ) 

npts = 10
model_params_dict = { 'or_sd': np.linspace(0.1, np.pi, npts), 'or_mean':np.linspace(0, np.pi, npts), 
                     'cur_mean' : np.linspace(-0.5, 1, npts), 'cur_sd': np.linspace(0.1, 1, npts)}   

model_params_dict = dm.cartesian_prod_dicts_lists( model_params_dict )
    
model_resp = apc_models( shape_dict_list = shape_dict_list, model_params_dict = model_params_dict)
dam =xr.DataArray(model_resp, dims = ['shapes', 'models'])
ds = xr.Dataset({'resp': dam})
ds.to_netcdf('/Users/dean/Desktop/net_code/responses/test_models_cdf.nc')
del model_resp, ds


dm = xr.open_dataset('/Users/dean/Desktop/net_code/responses/test_models_cdf.nc', chunks={'models': 100})
da = xr.open_dataset('/Users/dean/Desktop/net_code/responses/test_cdf.nc', chunks={'unit': 200})

#da = da.sel(x = 0, method = 'nearest' )
da_n = da['resp']/np.sqrt( (da['resp']**2).sum('shapes'))

b = (da_n*dm).sum('shapes')
fits = b.max('models')
fits = fits.max('x')
#ds = xr.Dataset({'r': c})
fits.to_netcdf('/Users/dean/Desktop/net_code/responses/test_r_cdf.nc')
#c = xr.open_dataset('/Users/dean/Desktop/net_code/responses/test_r_cdf.nc', chunks={'models': 1000})
 
    
    
    
    
    
    
    
##adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321] 
#a = np.hstack((range(14), range(18,318)))
#a = np.hstack((a, range(322, 370)))
#s = s[a]
#    
#
#nStim = np.size(s,0) 
#
#angularPosition = []
#curvature = []
#paramLens = []
#
#for shapeInd in range(nStim):
#    angularPosition.append(s[shapeInd][:, 0])
#    curvature.append(s[shapeInd][:, 1])
#    paramLens.append(np.size(s[shapeInd],0))
#    
#angularPosition = np.array(list(itertools.chain.from_iterable(angularPosition)))
#angularPosition.shape = (np.size(angularPosition),1)
#
#curvature = np.array(list(itertools.chain.from_iterable(curvature)))
#curvature.shape = (np.size(curvature),1)
#
##variable section length striding
#inds = np.empty((2,np.size(paramLens)),dtype = np.intp)
#inds[1,:] = np.cumsum(np.array(paramLens), dtype = np.intp) #ending index
#inds[0,:] = np.concatenate(([0,], inds[1,:-1])) #beginning index
#
#
#
#
##    #the Nonlin fit model for Pasupathy V4 Neurons
##    mat = l.loadmat('V4_370PC2001_LSQnonlin.mat')
##    f = np.array(mat['fI'][0])[0]
##    # orientation, curvature, orientation SD , curvature SD , correlation
##    
##    #use these to generate parameters for brute force model
##    maxAngSD = np.percentile(f[:,2], 100 - perc)
##    minAngSD = np.percentile(f[:,2], perc)
##    maxCurSD = np.percentile(f[:,3], 100 - perc)
##    minCurSD = np.percentile(f[:,3], perc)
#
#maxAngSD = np.deg2rad(171)
#minAngSD = np.deg2rad(23)
#maxCurSD = 0.98
#minCurSD = 0.09
#
##make this into a pyramid based on d-prime
#orMeans = np.linspace(0, 2*pi-2*pi/nMeans, nMeans) 
#orSDs = np.logspace(np.log10(minAngSD),  np.log10(maxAngSD),  nSD)
#curvMeans = np.linspace(-0.5,1,nMeans)
#curvSDs = np.logspace(np.log10(minCurSD),  np.log10(maxCurSD),  nSD)
#modelParams = cartesian([orMeans,curvMeans,orSDs,curvSDs])
#nModels = np.size( modelParams, 0)
