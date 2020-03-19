# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:58:01 2017

@author: deanpospisil
"""
layer = da.coords['layer_label'].values==b'conv1'
for unit in da[:, :, layer]:
    print(unit)