# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:19:03 2015

@author: dean
"""
import matplotlib.pyplot as plt
import numpy as np
inputLayer = np.random.randn(1000,3)

kernels = [2,2,1]
nLayers = len(kernels)

layers = []
for layer in range(nLayers):
    if layer == 0 :
        layers.append( np.random.randn(np.size(inputLayer,1), kernels[layer] ))
    else:
        layers.append( np.random.randn(kernels[layer-1], kernels[layer] ))
        
output = []
#first forward pass to initialize output list
for layer in range(nLayers):

    if layer == 0:
        output.append(np.dot( inputLayer, layers[0]))
    else:    
        output.append(np.dot( output[layer-1], layers[layer] ))
        
for layer in range(nLayers):
     
    if layer == 0:
        output[layer] = np.dot( inputLayer, layers[0])
    else:    
        output[layer] = np.dot( output[layer-1], layers[layer] )
        
    #now do the back propagation


#now, lets get into forward backward pass loops

#n training iterations

#forward pass loop
output = []
for layer in range(nLayers):
     
    if layer == 0:
        output[layer] = np.dot( inputLayer, layers[0])
    else:    
        output[layer] = np.dot( output[layer-1], layers[layer] )

#backward pass loop
for layer in range(nLayers)[-1::-1]:
    
    if layer == nLayers-1:
        delta[layer] = np.dot( inputLayer, layers[0])
    else:    
        delta[layer-1] = np.dot( output[layer-1], layers[layer] )    
    
    
plt.hist(output[-1], bins = 100)