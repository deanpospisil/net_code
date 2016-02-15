# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:23:13 2015

@author: dean
"""


r = np.array([0, 0, 1])
n = len(r)

nu=np.sum(r/np.double(n))**2
de=np.sum((r**2)/np.double(n))

print nu/de