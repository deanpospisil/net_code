# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:53:34 2016

@author: deanpospisil
"""
import numpy as np
a = np.random.randn(10,20)

u, s, v = np.linalg.svd(a, full_matrices=False)

print(np.sum(s**2))
print(np.linalg.norm(a))
#x, res,ran, s = np.linalg.lstsq(u[:2,:], a)