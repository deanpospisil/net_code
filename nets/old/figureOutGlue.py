# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:24:53 2015

@author: dean
"""

import numpy as np
import pandas as pd

x = [1, 2, 3]
y = [2, 3, 4]

u = [10, 20, 30, 40]
v = [20, 40, 60, 80]

pandas_data = pd.DataFrame({'x': x, 'y': y})
dict_data = {'u': u, 'v': v}
recarray_data = np.rec.array([(0, 1), (2, 3)],
                             dtype=[('a', 'i'), ('b', 'i')])