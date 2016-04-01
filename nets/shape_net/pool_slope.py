# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 13:56:08 2016

@author: dean
"""

image_size = 64.

while image_size>2:
    pad = 0.
    stride = 2.
    ks = 3.
    image_size = ((image_size +2*pad - ks)/stride + 1)
    print image_size
