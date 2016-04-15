# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:05:22 2016

@author: deanpospisil
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
import husl

def ziphusl(a):
    rgb = husl.huslp_to_rgb(a[0], a[1], a[2])
    return rgb
    
hue = np.ones((1, 100)) * np.linspace(0,1,100).reshape(100,1)*360
light = np.ones((1, 100)) * np.linspace(0,1,100).reshape(100,1)*80
light = light.T
sat = np.ones((100, 100))*100

hsl = np.dstack((hue, sat, light))
rgb = np.apply_along_axis(ziphusl, 2, hsl)

plt.imshow(rgb, interpolation = 'nearest')
plt.xticks(range(0,100,25))
plt.yticks(range(0,100,25))
plt.gca().set_xticklabels(['0', .25, .5, .75, 1])
plt.gca().set_yticklabels(reversed(['0', 90, 180, 270, 360]))

plt.scatter(range(0,100), range(0,100))