# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:28:17 2015

@author: dean
"""
import matplotlib.pyplot as plt
import time
plt.close('all')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

import numpy as np

#desiredOutput = 100
#theInput = np.array([1, -1, 1])
#learningRate = 0.1
#
#theWeights = np.array([ 0.1, 0.3, 5 ])
#
#for ind in range(10000):
#    theOutput = np.sum( theWeights * theInput )
#    
#    energy = 0.5 * (desiredOutput - theOutput)**2
#    #print energy
#    
#    deltas = (desiredOutput - np.sum(theWeights * theInput)) * theInput
#    
#    
#    theWeights = learningRate*deltas + theWeights
#    
#lets add in another layer
    

w11 = np.random.randn(3)/20
w12 = np.random.randn(3)/20
w21 = np.random.randn(2)/20
np.random.randn()/20
theInput = np.random.randn(3)/20
theInput = np.array([0.5,1,0.5])/10.
desiredOutput =1
learningRate = 0.1
energy = []
plotCount = 0
for ind in range(1000):
    
    # the input dot product with 
    
    o11 = np.dot( w11, theInput)
    o12 = np.dot( w12, theInput)
    
    o1 = np.array([o11, o12])
    o2 = np.dot(o1,w21)
    
    energy.append((desiredOutput - o2))


    
    
    d2 = (desiredOutput - o2) * o1 
    
    w21 = d2 * learningRate + w21
    
    
    d11 = ( (desiredOutput - o2)*w21[0]  ) * theInput
    d12 = ( (desiredOutput - o2)*w21[1]  ) * theInput
    
    w11 = d11*learningRate + w11
    w12 = d12*learningRate + w12
    plotCount+=1
    if plotCount>100:
        scale = 0.1
        ax.set_xlim([-scale,scale])
        ax.set_ylim([-scale,scale])    
        ax.set_zlim([-scale,scale])       
        ax.plot([0, w11[0]], [0, w11[1]], [0, w11[2]], color='r')
        plt.show()
        
    
        ax.plot([0, w12[0]], [0, w12[1]], [0, w12[2]], color='g')
        print energy[-1]
        ax.plot([0, theInput[0]], [0, theInput[1]], [0, theInput[2]], color='b')
        plt.show()
        

        ax2.plot([0,o11], [0,o12], color = 'b')
        ax2.plot([0,w21[0]], [0,w21[1]],color='r')
        scale2=0.5
        ax2.axis([-scale2,scale2,-scale2,scale2])
        plt.show()
        plt.pause(1)
        plotCount=0

    
 


#plt.plot(energy)
#plt.axis([0, 1000, -desiredOutput, desiredOutput])