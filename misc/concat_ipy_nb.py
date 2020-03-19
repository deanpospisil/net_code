# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:47:25 2017

@author: deanpospisil
"""

import json
import os
#!/usr/bin/env python
# Note, updated version of 
# https://github.com/ipython/ipython-in-depth/blob/master/tools/nbmerge.py
"""
usage:
python nbmerge.py A.ipynb B.ipynb C.ipynb > merged.ipynb
"""

import io
import os
import sys

import json
from IPython.nbformat import current

def merge_notebooks(filenames):
    merged = None
    for fname in filenames:
        with io.open(fname, 'r', encoding='utf-8') as f:
            nb = current.read(f, 'json')
        if merged is None:
            merged = nb
        else:
            merged.worksheets[0].cells.extend(nb.worksheets[0].cells)
    merged.metadata.name += "_merged"
    return merged

    
folder = '/Users/deanpospisil/Desktop/modules/ipython_nb'
prefix = 'pt'
paths = [os.path.join(folder, name) for name in os.listdir(folder) if name.startswith(prefix) and name.endswith(".ipynb")]
merged = merge_notebooks(paths)
