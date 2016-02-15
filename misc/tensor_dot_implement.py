# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:28:31 2016

@author: deanpospisil
"""


import xarray as xr

from xarray import align, DataArray
# note: using private imports (e.g., from xarray.core) is definitely discouraged!
# this is not guaranteed to work in future versions of xarray
from xarray.core.ops import _dask_or_eager_func

def tensordot(a, b, dims):
    if not (isinstance(a, DataArray) and isinstance(b, DataArray)):
        raise ValueError

    a, b = align(a, b, join='inner', copy=False)

    axes = (a.get_axis_num(dims), b.get_axis_num(dims))
    f = _dask_or_eager_func('tensordot', n_array_args=2)
    new_data = f(a.data, b.data, axes=axes)

    if isinstance(dims, str):
        dims = [dims]

    new_coords = a.coords.merge(b.coords).drop(dims)

    new_dims = ([d for d in a.dims if d not in dims] +
                [d for d in b.dims if d not in dims])

    return DataArray(new_data, new_coords, new_dims)
    
  
import numpy as np


x_trans = np.linspace(-3,3,6)
y_trans = np.linspace(-3,3,5)
imgID = range(4)
da = xr.DataArray( np.ones((6,5,4)), 
                  coords = [ x_trans, y_trans, imgID ], 
                  dims = ['x_trans', 'y_trans', 'imgID'] )


models = range(20)
dm = xr.DataArray( np.ones(( 20 , 5, 4 )), 
                  coords = [  models, y_trans, imgID], 
                  dims = [  'models', 'y_trans', 'imgID'  ] )                                  

#xarray tensordot
proj_a = tensordot(da, dm, 'imgID')

#dask xarray tensor dot
da = da.chunk()
dm = dm.chunk()
proj_b = tensordot(da, dm, 'imgID')


##errors
#proj_c = tensordot(da, dm, ['imgID', 'y_trans'])
#
#da = da.chunk()
dm = dm.load()
#proj_d = tensordot(da, dm, 'imgID')






