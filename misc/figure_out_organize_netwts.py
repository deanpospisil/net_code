# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:12:06 2016

@author: deanpospisil
"""

import pickle
import sys  
 
with open('/Users/deanpospisil/Desktop/modules/net_code/nets/netwts.p', 'rb') as f:
    a = pickle.load(f, encoding='latin1')