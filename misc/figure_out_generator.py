# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:54:35 2016

@author: deanpospisil
"""
import itertools
a = [1,2,3]
g = (x**2 for x in a )

gs=itertools.islice(g,2)
print(gs.__next__())
print(gs.__next__())