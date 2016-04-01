# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:58:22 2016

@author: dean
"""



import caffe
caffe.set_mode_gpu()
net = caffe.Net(ann_dir + 'deploy.prototxt', ann_dir + ann_fn + '.caffemodel', caffe.TEST)
#net will have some set of weights, need to adjust weights and shapes to whatever my new
#net architecture will look like.
net_full_conv.save('net_surgery/bvlc_caffenet_full_conv.caffemodel')