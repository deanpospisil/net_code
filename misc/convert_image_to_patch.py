# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:13:31 2016

@author: deanpospisil
"""

#generate and shuffle patches
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from scipy import io
fname = 'tofu.psych.upenn.edu/zip_etjrgupovc/'
img_dir = top_dir + 'data/responses/' + fname
img_dir = '/Users/deanpospisil/Desktop/wget-1.5.3/tofu.psych.upenn.edu/zip_etjrgupovc/cd01A/'
img_names = os.listdir(img_dir)
rgb_ims = [name for name in img_names if 'RGB.mat' in name]
image = [io.loadmat( img_dir + name , squeeze_me=True)['RGB_Image'] for name in rgb_ims[:3]]

patch_size = 10
data = extract_patches_2d(np.concatenate(image[:1],0), (patch_size, patch_size))
window = np.hamming(patch_size).reshape(1,patch_size) * np.hamming(patch_size).reshape(patch_size,1)
window = window.reshape(1, patch_size, patch_size, 1)
data_w = data*window
plt.figure()
plt.imshow(data[0], interpolation='none')
plt.figure()
plt.imshow(data_w[0],interpolation='none'
  )



#data -= np.mean(data, axis=0)
#data /= np.std(data, axis=0)
