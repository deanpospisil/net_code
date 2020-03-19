import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + 'v4cnn'

def ti_av_cov(da):
    dims = da.coords.dims
    #get the da in the right shape
    if ('x' in dims) and ('y' in dims):
        da = da.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        da = da.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        da = da.transpose('unit', 'shapes', 'y')
        
    #some data to store
    ti = np.zeros(np.shape(da)[0])
    dens = np.zeros(np.shape(da)[0])
    nums = np.zeros(np.shape(da)[0])
    tot_vars = np.zeros(np.shape(da)[0])
    kurt_shapes = np.zeros(np.shape(da)[0])
    kurt_x =  np.zeros(np.shape(da)[0])

    for i, unit_resp in enumerate(da):
        if len(unit_resp.shape)>2:
            #unwrap spatial
            unit_resp = unit_resp.values.reshape(unit_resp.shape[0], unit_resp.shape[1]*unit_resp.shape[2])   
        else:
            unit_resp = unit_resp.values
        unit_resp = unit_resp.astype(np.float64)
        unit_resp = unit_resp - np.mean(unit_resp, 0, keepdims=True, dtype=np.float64)
 

        cov = np.dot(unit_resp.T, unit_resp)
        cov[np.diag_indices_from(cov)] = 0
        numerator = np.sum(np.triu(cov))

        vlength = np.linalg.norm(unit_resp, axis=0, keepdims=True)
        max_cov = np.outer(vlength.T, vlength)
        max_cov[np.diag_indices_from(max_cov)] = 0
        denominator= np.sum(np.triu(max_cov))

        kurt_shapes[i] = kurtosis(np.sum(unit_resp**2, 1))
        kurt_x[i] = kurtosis(np.sum(unit_resp**2, 0))
        den = np.sum(max_cov)
        num = np.sum(cov)
        dens[i] = den
        nums[i] = num
        tot_vars[i] = np.sum(unit_resp**2)
        if den!=0 and num!=0:
            ti[i] = num/den 
    return ti, kurt_shapes, kurt_x, dens, nums, tot_vars 

#
    
#x = np.zeros((22096, 300, 400))
def norm_av_cov(x):
    x = x.astype(np.float64)
    
    cov = np.matmul(np.transpose(x, axes=(0, 2, 1)), x)
    numerator = np.sum(np.triu(cov, k=1), (1, 2))
    
    vlength = np.linalg.norm(x, axis=1, keepdims=True)
    max_cov = np.multiply(np.transpose(vlength, axes=(0, 2, 1)), vlength)
    denominator= np.sum(np.triu(max_cov, k=1), (1, 2))
    
    norm_cov = numerator/denominator
    return norm_cov


import pickle
goforit=False       
if 'netwts' not in locals() or goforit:
    with open(top_dir + '/nets/netwts.p', 'rb') as f:    
        try:
            netwts = pickle.load(f, encoding='latin1')
        except:
            netwts = pickle.load(f)
# reshape fc layer to be spatial
netwts[5][1] = netwts[5][1].reshape((4096, 256, 6, 6))
wts_by_layer = [layer[1] for layer in netwts]   

net_resp_name = 'bvlc_reference_caffenety_test_APC362_pix_width[32.0]_x_(104.0, 124.0, 11)_x_(104.0, 124.0, 11)_amp_None.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
if not type(da.coords['layer_label'].values[0]) == str:
    da.coords['layer_label'].values = [thing.decode('UTF-8') for thing in da.coords['layer_label'].values]
da.coords['unit'] = range(da.shape[-1])

from more_itertools import unique_everseen
layer_num = da.coords['layer']
layer_label_ind = da.coords['layer_label'].values
split_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',]
dims = ['unit','chan', 'y', 'x']
layer_names = list(unique_everseen(layer_label_ind))
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6',]


netwtsd = {}
for layer, name in zip(wts_by_layer, layer_names):
    dim_names = dims[:len(layer.shape)]
    layer_ind = da.coords['layer_label'].values == name 
    _ =  da[..., layer_ind].coords['unit']
    netwtsd[name] = xr.DataArray(layer, dims=dims, 
           coords=[range(n) for n in np.shape(layer)])
    netwtsd[name].coords['unit'] = _ 

    
    
    