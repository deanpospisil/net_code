# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:05:15 2016

@author: deanpospisil
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sample_rate_mult=1
ft = np.zeros((20,11))
ft[7, 0] = 10
img = np.fft.irfft2(ft)

ft = np.fft.fft2(img)
ftr = np.fft.rfft2(img)
plt.subplot(311)
plt.imshow(abs(ft), interpolation = 'none',cmap = cm.Greys_r)
plt.subplot(312)
plt.imshow(abs(ftr), interpolation = 'none',cmap = cm.Greys_r)
plt.subplot(313)
plt.imshow(img, interpolation = 'none',cmap = cm.Greys_r)
