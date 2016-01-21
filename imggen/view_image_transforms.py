# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:07:09 2016

@author: dean
"""


import os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)
    


import matplotlib.pyplot as plt
import d_img_process as imp 
import matplotlib.cm as cm
    
    
img_dir =  '/Users/dean/Desktop/net_code/images/baseimgs/PC370/'  
img_dir =  '/Users/deanpospisil/Desktop/net_code/images/baseimgs/PC370/'  
stack, stack_desc = imp.load_npy_img_dirs_into_stack( img_dir )

plt.imshow(stack[1,:,:])
trans_stack = imp.imgStackTransform( {'shapes':[2,2],'blur':[1, 0.25]}, stack )

plt.imshow(trans_stack[0,:,:],cmap = cm.Greys_r, interpolation = 'none')