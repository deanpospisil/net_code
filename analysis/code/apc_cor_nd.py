# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:54:30 2016

@author: dean
"""

import os, sys
import numpy as np
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)s
sys.path.append( top_dir + '/xarray')

import xarray as xr
import d_misc as dm


def cor_resp_to_model(da, dmod, fit_over_dims=None):
    #typically takes da, data, and dm, a set of linear models, an fn to write to,
    #and finally fit_over_dims which says over what dims is a models fit supposed to hold.

    da = da - da.mean(('shapes'))
    dmod = dmod['resp']

    resp_n = da.vnorm(('shapes'))
    proj_resp_on_model = da.dot(dmod)

    if not fit_over_dims == None:
        resp_norm = resp_n.vnorm(fit_over_dims)
        proj_resp_on_model_var = proj_resp_on_model.sum(fit_over_dims)
        n_over = 0
        #count up how many unit vectors you'll be applying for each r.
        for dim in fit_over_dims:
            n_over = n_over + len(da.coords[dim].values)
    else:
        resp_norm =  resp_n
        proj_resp_on_model_var = proj_resp_on_model
        n_over = 1

    all_cor = (proj_resp_on_model_var) / (resp_norm * (n_over**0.5))
    all_cor = all_cor.load()

    corarg = all_cor.argmax('models', skipna=True)
    model_fit_params = dmod.coords['models'][corarg]
    cor = all_cor.max('models')

    for key in model_fit_params.coords.keys():
        cor[key] = ('unit', np.squeeze(model_fit_params[key]))

    sha = dm.provenance_commit(top_dir)
    cor.attrs['analysis'] = sha
    cor.attrs['model'] = dmod.attrs['model']

    return cor


dmod = xr.open_dataset(top_dir + 'analysis/data/models/apc_models.nc',
                       chunks = {'models': 1000, 'shapes': 370}  )

da = xr.open_dataset( top_dir + 'analysis/data/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', chunks={'unit': 100})
da = da.sel(x=0, method='nearest')
da = da.sel(unit = range(109),  method = 'nearest')
cor = cor_resp_to_model(da['resp'], dmod, fit_over_dims = None)

'''
#da = xr.open_dataset( top_dir + 'analysis/data/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc', chunks = {'unit': 100}  )
<<<<<<< HEAD
dmod = dmod.sel(models = range(10), method = 'nearest' )
ds = xr.open_mfdataset(top_dir + 'analysis/data/iter_*.nc', concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
da = ds['resp'].chunk(chunks = {'niter':1, 'unit':100, 'shapes':370})
da = da.sel(x = np.linspace(-50, 50, 51), method = 'nearest' )
#da = da.sel(niters=np.linspace(0, da.coords['niter'].shape[0], 19),  method = 'nearest')
=======

#dmod = dmod.sel(models = range(10), method = 'nearest' )
#ds = xr.open_mfdataset(top_dir + 'analysis/data/iter_*.nc',
#                       concat_dim = 'niter', chunks = {'unit':100, 'shapes': 370})
#da = ds.to_array().chunk(chunks = {'niter':1, 'unit':100, 'shapes': 370})
#da = da.sel(x = np.linspace(-50, 50, 2), method = 'nearest' )
#da = da.sel(niter = np.linspace(0, da.coords['niter'].shape[0], 2),
#                                method = 'nearest')
>>>>>>> 6789ab6c1a4d6d1d23bf50a0e8d263b6b39de031
#da = da.sel(unit = range(10),  method = 'nearest')


for iterind in ds.niter.values:
    da_c = da.sel(niter=iterind)
    cor = cor_resp_to_model(da_c, dmod, fit_over_dims = ('x',))
    cor.to_dataset(name='r').to_netcdf(top_dir + 'analysis/data/r_iter_' + str(iterind) + '.nc')

ds = xr.open_mfdataset(top_dir + 'analysis/data/r_iter_*.nc', concat_dim = 'niter')
ds.to_netcdf(top_dir + 'analysis/data/r_iter_total_' + str(da.niter.shape[0]) +  '.nc')
'''
