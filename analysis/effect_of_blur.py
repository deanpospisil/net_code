# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:21:03 2016

@author: dean
"""
import xray as xr
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)

## effect of blur
#dm = xr.open_dataset(cwd +'/responses/apc_models.nc', chunks={'models': 100} )
#da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_blur_0.1_2.0_10.nc', chunks={'blur': 1, 'unit': 100} )
#da = xr.open_dataset(cwd +'/responses/PC370_shapes_0.0_369.0_370_blur_0.25_1000.0_5.nc', chunks={'blur': 1, 'unit': 100} )
##
#da['resp'].chunk()
#unit_sel = np.arange(1, f.dims['unit'],1000 )
##da = da.sel(unit = unit_sel, method = 'nearest' )

f1 = xr.open_dataset(cwd + '/responses/apc_models_r_blur.nc' )

s=[]
labels = f1.coords['layer_label'].values
for label in labels:
    if not label in s:
        s.append(label)
        
layer = 12
unitsel = np.arange(0, f1.dims['unit'],1 )

f1 = f1.sel(unit = unitsel, method = 'nearest' )
b = f1.to_dataframe()

#b.set_index(['layer_unit' ,'layer', 'layer_label'], append=True, inplace=True)
#b.reset_index(['layer_unit' ,'layer', 'unit'],drop=True, inplace=True)
b.reset_index([ 'blur'], inplace=True)
plt.close('all')
sns.boxplot(x="layer_label", y="resp" ,hue='blur', data=b[b['resp']>0], whis=1)

#f = f1.sel(unit = f1.groupby('layer_label').group == s[layer], method = 'nearest' )
#
#
#a = list(f.groupby('layer_label'))
#
#labels = [l[0] for l in a]
#
#ord_arr = []
#for o_label in s: 
#    for i,label in enumerate(labels):
#        if label == o_label:
#            ord_arr.append(a[i])
#            
#plt.close('all')
#(f['resp'].to_pandas()>0.5).T.mean().plot(kind='bar')
#plt.title(s[layer])
#plt.tight_layout()
#
#
