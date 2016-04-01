# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:39:07 2016

@author: dean
"""
import os
import pickle
import numpy as np
top_dir = os.getcwd().split('net_code')[0] + '/net_code/'

s = ['conv1', 'conv2', 'conv3', 'conv4',  'conv5',  'fc6', 'fc7', 'fc8'   ]
an = pickle.load(open( top_dir + "nets/alexNetWeights.p", "rb") )
f = open('anw.txt', 'w')

for i, layer in enumerate(an[0]):

    f.write('LAYER ' + s[i] + ' weights ')
    for dim in np.shape(layer):
        f.write(str(dim) + ' ')
    f.write('\n')
    for w in layer.flatten():
        f.write(str(w)+ ' ')
    f.write('\n')

    bvec = an[1][i]
    f.write( 'LAYER ' + s[i] + ' biases ')
    for dim in np.shape(bvec):
        f.write(str(dim) + ' ')
    f.write('\n')
    for b in bvec.flatten():
        f.write(str(b) + ' ')
    f.write('\n')
    if i>1:
        break

f.close()

#conv1weights (96, 3, 11, 11) -0.00676263 0.0169465
#conv1biases (96,) 0.283837 -0.499682
#last index changes fastest

'''
for i, layer in enumerate(an[0]):
    plt.subplot(8, 1, i+1)
    plt.hist(layer.flatten(), bins = 100)

    ax = plt.gca()
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.linspace(start, end, 2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    plt.title(s[i])
    plt.tight_layout()
    print(i)

'''

