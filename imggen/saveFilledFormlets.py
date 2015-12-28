# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:13:27 2015

@author: dean
"""
import applyGaborFormlet
import os,sys
import Image


plt.figure(figsize=(10,10),interpolation='None')
points = array([real(intcShape), imag(intcShape)]).T
line = plt.Polygon(points, closed=True, fill='k', edgecolor='none',fc='k')
plt.gca().add_patch(line)
plt.axis('off')
plt.gca().set_xlim([-radius*2, radius*2])
plt.gca().set_ylim([-radius*2, radius*2])
plt.savefig('/home/dean/Desktop/AlexNet_APC_Analysis/shape.png',bbox_inches=None)



img = Image.open('/home/dean/Desktop/AlexNet_APC_Analysis/shape.png')
ima=array(img)[:,:,1]
imshow(ima,interpolation='none')