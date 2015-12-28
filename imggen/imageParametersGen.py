# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:09:07 2015

@author: dean
"""

# parameter generation
import numpy as np
import pickle
import itertools
import os
import shutil


baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
#these are the transformations you will make
baseStim = baseImageList[0]


stochastic = True
if stochastic is False:
    stimType = 'cart'
else:
    stimType = 'stoch'
    
genStim = baseStim + '_' + stimType  
    
baseStimDir = '/Users/dean/Desktop/shapenet/stim/basestim/'
transStimDir = '/Users/dean/Desktop/shapenet/stim/trans_stim/'

os.chdir(baseStimDir+baseStim)
files = os.listdir(os.getcwd())

nbaseImgs=0
for fnm in files:
    if '.npy' in fnm:
        nbaseImgs+=1

transNames = ['shape', 'x','y','scale', 'rot' ]
transNamesSubset = ['shape', 'x', 'y', 'scale', 'rot'] 
description = genStim

paramRangeDict = {}

if stochastic:
    nSamplings = 5
    nImgs = nbaseImgs*nSamplings
    
    #start and stop are the range of the uniform dist, or the range of the STD's
    dist = ['lin', 'uni', 'uni', 'uni', 'uni']
    start = [0,          -5, -5, 1, 0 ]
    stop =  [nbaseImgs-1, 5,  5, 1, 360. ]
    
    for i, name in enumerate(zip(transNames)):
        if name[0] in transNamesSubset:
            paramRangeDict[name[0]] = [ dist[i], start[i], stop[i] ]
            description = description + '_' + name[0] + '_'+  str(int(start[i]))  + '_'+  str(int(stop[i])) + '_' +  dist[i]
    
    transPts = np.zeros((nImgs, len(transNamesSubset)))
    ind = 0
    for name in transNamesSubset:
        if name is  'shape':
            transPts[:,0]= np.tile(np.arange(0,nbaseImgs), (nSamplings,1)).ravel('F')
        else:
            ind +=1
            curDist = paramRangeDict[name][0]

            if curDist is 'uni':
                transPts[:,ind] = np.random.uniform(paramRangeDict[name][1], paramRangeDict[name][2], nImgs)
            elif curDist is 'gau':
                'to do'

          
    

else:

    npts =  [nbaseImgs,   2,  2, 1, 2 ]
    start = [0,          -5, -5, 1, 0 ]
    stop =  [nbaseImgs-1, 5,  5, 1, 360. - 360. / np.double(npts[4]) ]
    
    
    #just pull parameters from those in subset
    for i, name in enumerate(zip(transNames)):
        if name[0] in transNamesSubset:
            paramRangeDict[name[0]] = [ npts[i], start[i], stop[i] ]
            description = description + '_' + name[0] + '_'+  str(int(start[i]))  + '_'+  str(int(stop[i])) + '_' +  str(int(npts[i])) 
            
    
    
    # enumerate all the points in your ranges
    transPts = []
    for name in transNamesSubset:
        transPts.append(np.linspace( paramRangeDict[name][1], paramRangeDict[name][2], paramRangeDict[name][0] ))
    
    
    #get the cartesian product of these for all possible tuples
    transPts = np.array(list(itertools.product(*transPts)))
    
    

nImgs = np.size(transPts,0)

#make a dict with these.
transDict = {}
for i, name in enumerate(zip(transNamesSubset)):
    transDict[name[0]] = transPts[ :, i ]  


        
        
#put this        
os.chdir(transStimDir)
try:
    os.mkdir(genStim)

except:
    shutil.rmtree(genStim)
    os.mkdir(genStim)


os.chdir(transStimDir + genStim)

pickle.dump( transDict, open( description +'.p', "wb" ) )