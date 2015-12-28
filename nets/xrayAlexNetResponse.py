# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:10:23 2015

@author: dean
a"""

import sys
import os
caffe_root = '/home/dean/caffe/'
sys.path.insert(0, caffe_root + 'python')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
cwd = os.path.dirname(dname)
sys.path.append( cwd)

import dMisc as dm
import dImgProcess as imp
import numpy as np
import xray as xr
import collections
import itertools
import dMisc as misc
import copy
from sklearn.utils.extmath import cartesian

transNames = ['shape', 'x', 'scale' ]
shape = np.arange(370, dtype =np.float64) 
scale = np. linspace(0.1,1,21)
theEnd = 70
npts = 10

x = list(np.linspace(-theEnd,theEnd,npts))
y = list(np.linspace(-theEnd,theEnd,npts))
scale = list(np.linspace(0.1,1,5))
param_v = []

param_v.append(shape)
param_v.append(x)
param_v.append(scale)

#param_v.append(y)
params = collections.OrderedDict()
for ind in range(len(param_v)):
    params[transNames[ind]] = param_v[ind]


l_params = tuple([len(value) for key, value in params.items()])

 
resp = xr.DataArray(np.empty(l_params), params)

#this will be used as a template for when it is expanded for all units
resp.values[:] = np.nan


#so now, how do I pull out the coordinates for making all the stimuli?
keys=list(transNames) 
valList = [resp.coords[name].values for name in keys]   
indList = [range(len(resp.coords[name].values)) for name in keys] 
nParams = len(valList)

indDict = collections.OrderedDict()


#make the parmeters list, and the indices list
imgParam = []
imgParam = np.array(list(itertools.product(*valList)))
imgParamInd = np.array(list(itertools.product(*indList)))

nImgs = np.size(imgParam,0)


ANNDir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ANNFileName='bvlc_reference_caffenet.caffemodel'
source='AlexNet'
stim = stimName= '370PC2001Scale'

for ind in range(len(transNames)):
    stim = stim +'_'+ transNames[ind] + str(len(param_v[ind]))

respDir= cwd +'/' + 'responses/'
responseFile = respDir + source  + '_'  + stim
dm.ifNoDirMakeDir(respDir)


baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseImage = baseImageList[0] 
imageDir = cwd +'/' + 'images/baseimgs/' + baseImage + '/'


stackSize=200
saveImgList = [False]* nImgs
saveImgList[0:stackSize] =  [True] * (stackSize+1)
defaultStackSize = stackSize

#these will be the indices for pulling out stacks
#remainder tells you the length of the last bit if there was a remainder
stackInd, remainder = misc.sectStrideInds(stackSize, nImgs)
nPass = np.size(stackInd,0)

import caffe

caffe.set_mode_gpu()
net = caffe.Net(
    ANNDir+'deploy.prototxt',
    ANNDir+ANNFileName, 
    caffe.TEST)
net.blobs['data'].reshape(defaultStackSize, 3, 227, 227)    

# preload image directory
imgIDs = resp.coords['shape'].values
imgInd = 0
shapeSet = np.zeros((np.size( imgIDs ), 227, 227))

for imgName in imgIDs :
    fnm = imageDir + str(int(imgName)) + '.npy'
    shapeSet[ imgInd, :, : ]  = np.load(fnm)
    imgInd+=1

#get the layerNames, units for each layer, and how many layers
layerNameList = [(k) for k, v in net.params.items()]
nUnits = [(v[0].data.shape[0]) for k, v in net.params.items()]
nLayers = np.shape(layerNameList)[0]

#make as many data arrays as their are layers, with the right number of units
netResp = []
netRespArr = [ ]
paramsWithUnits = copy.deepcopy(params)
for ind in range(nLayers):
    shapeWithUnits = l_params + (nUnits[ind],)
    paramsWithUnits['unit'] = range(nUnits[ind])
    netRespArr.append(np.zeros(shapeWithUnits))
    netResp.append(xr.DataArray(np.empty(shapeWithUnits), paramsWithUnits))
    netResp[ind][:] = np.nan


paramInd = -1
stack = np.empty( ( defaultStackSize, 3, 227, 227) )
stack[:] = np.NAN
stackSize = defaultStackSize
  

#this is each time you pass a stack of images through the net
for passInd in range(nPass):
        
    beg = stackInd[passInd,0 ]
    fin = stackInd[passInd,1 ]
    
    #load the ind Dict
    imgParam_sect = imgParam[ beg:fin , : ]
    imgParamInd_sect = imgParamInd[ beg:fin , : ]
    for key, i in zip(keys, range(len(keys))):
        indDict[key] = imgParam_sect[ : , i ]    
    
    #fill the shape set with the initial shapes
    trans_stack = shapeSet[np.intp(indDict['shape']),:,: ]    
    stackHeight = np.size( trans_stack, 0 )
    net.blobs['data'].reshape( stackHeight, 3, 227, 227 )
    
    #transform the stack according to the stack of params in indDict
    trans_stack = imp.imgStackTransform( indDict, trans_stack )
    stack = np.tile(trans_stack, (3,1,1,1))
    stack = np.swapaxes(stack, 0, 1)
    
    #save all th images ofr testing 
    paramInd+=1
    if saveImgList[paramInd]:
        for saveInd in range(stackHeight):
            
            imp.saveToPNGDir(imageDir + stimName + '/' , str(str(imgParam_sect[saveInd,:])), stack[saveInd,:,:,:])

    print(passInd/np.double(nPass))
    
    #run the full stack through the net    
    net.blobs['data'].data[...] = stack
    net.forward()
    
    
    
    for layerInd, layerName in zip( range(len(layerNameList)), layerNameList):

        print(layerName)
        #by image sequentially fill all units responses to that image
        if layerName[0] == 'c':
                        
            
            mid = round(np.shape(net.blobs[layerName].data)[3] / 2)
            blob_inds = cartesian(( range(stackHeight) , range(nUnits[layerInd]), [mid,], [mid,] ))
            resp_inds =  np.hstack(( imgParamInd_sect[blob_inds[ : , 0],:] , np.reshape(blob_inds[ : , 1], (len(blob_inds),1)) ))
         
            
            netRespArr[layerInd][map(tuple, resp_inds.T)] = net.blobs[layerName].data[ map(tuple, blob_inds.T)]
                
             
        else:
            
            
            blob_inds = cartesian(( range(stackHeight) , range(nUnits[layerInd])))
            resp_inds =  np.hstack(( imgParamInd_sect[blob_inds[ : , 0],:] , np.reshape(blob_inds[ : , 1], (len(blob_inds),1)) ))
                  
            
            netRespArr[layerInd][map(tuple, resp_inds.T)] = net.blobs[layerName].data[map(tuple, blob_inds.T)]
        
netResp = []
paramsWithUnits = copy.deepcopy(params)
for layerInd in range(nLayers):
    
    paramsWithUnits['unit'] = range(nUnits[layerInd])
    netResp.append(xr.DataArray( netRespArr[layerInd], paramsWithUnits))



import pickle
with open( responseFile + '.pickle', 'w') as f:
    pickle.dump( [ netResp ] , f )

