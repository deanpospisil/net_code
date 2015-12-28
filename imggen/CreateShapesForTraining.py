

import dImgProcess as imp
import numpy as np
import sys
import os
import collections
import itertools
import dMisc as misc
import scipy as sc
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt


baseImageDir = '/Users/dean/Desktop/AlexNet_APC_Analysis/stim/basestim/PC370/'
saveImgDir =  '/Users/dean/Desktop/AlexNet_APC_Analysis/test_shape_save/'
#
#baseImageDir = '/home/dean/Desktop/AlexNet_ReceptiveField/PC370/'
#saveImgDir =  '/home/dean/caffe/shape/'

trainvaltestFolderNames = ['train', 'val', 'test']

# lets remove all previous images if there are any
for name in trainvaltestFolderNames:
    
    folder = saveImgDir +'/'+ name
    
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
    
        except Exception, e:
            print e

#these are the transformations you will make
transNames = ['shape', 'x','y' ]
shape = np.arange(370, dtype =np.float64) 
scale = np. linspace(0.1,1,21)
theEnd = 70
npts = 2

x = list(np.linspace(-theEnd,theEnd,npts))
y = list(np.linspace(-theEnd,theEnd,npts))

param_v = []
param_v.append(shape)
#param_v.append(scale)
param_v.append(x)
param_v.append(y)


valList = param_v  


stackSize=200
defaultStackSize = stackSize


#make the parmeters list, and the indices list with a cartesian product
imgParam = []
imgParam = np.array(list(itertools.product(*valList)))

nImgs = np.size(imgParam,0)


stim = stimName ='370PC2001'
for ind in range(len(transNames)):
    stim = stim +'_'+ transNames[ind] + str(len(param_v[ind]))

############ 
#randomly divide the images up into the three folders.
perc = np.arange(0,nImgs)/np.double(nImgs-1)
TrainValOrTest = np.zeros(np.size(perc), dtype = np.int)
TrainValOrTest[perc<0.5] = 0
TrainValOrTest[perc>0.5] = 1
TrainValOrTest[perc>0.75] = 2
np.random.shuffle(TrainValOrTest)

indDict = collections.OrderedDict()


#these will be the indices for pulling out stacks
#remainder tells you the length of the last bit if there was a remainder
stackInd, remainder = misc.sectStrideInds(stackSize, nImgs)
nPass = np.size(stackInd,0)


# preload image directory
imgIDs = shape
imgInd = 0
shapeSet = np.zeros((np.size( imgIDs ), 227, 227))

for imgName in imgIDs :
    fnm = baseImageDir + str(int(imgName)) + '.npy'
    shapeSet[ imgInd, :, : ]  = np.load(fnm)
    imgInd+=1



paramInd = -1
stack = np.empty( ( defaultStackSize, 3, 227, 227) )
stack[:] = np.NAN
stackSize = defaultStackSize
  
#nPass=1
#this is each time you pass a stack of images through the net
for passInd in range(nPass):
        
    beg = stackInd[passInd,0 ]
    fin = stackInd[passInd,1 ]
    
    #load the ind Dict
    imgParam_sect = imgParam[ beg:fin , : ]
    
    for key, i in zip(transNames, range(len(transNames))):
        indDict[key] = imgParam_sect[ : , i ]    
    
    #fill the shape set with the initial shapes
    trans_stack = shapeSet[np.intp(indDict['shape']),:,: ]    
    stackHeight = np.size( trans_stack, 0 )
    
    #transform the stack according to the stack of params in indDict
    trans_stack = imp.imgStackTransform( indDict, trans_stack )
    stack = np.tile(trans_stack, (3,1,1,1))
    stack = np.swapaxes(stack, 0, 1)
    
    #save all th images ofr testing 

    for saveInd in range(stackHeight):
        paramInd+=1
        shapeId = np.int(imgParam_sect[saveInd][0])
        fileName = str(paramInd) + '_' + str(shapeId) 
        sc.misc.imsave( saveImgDir + '/' + trainvaltestFolderNames[TrainValOrTest[paramInd]] + '/' + fileName + '.bmp', stack[saveInd,:,:,:])


    print(passInd/np.double(nPass))
    
    

os.chdir(saveImgDir)

for folder in trainvaltestFolderNames:
    
    print folder
    
    files = os.listdir( saveImgDir + folder )
    res = [(k + ' ' + k.split('_')[1].split('.')[0]) for k in files if 'bmp' in k]
    
    a = open( folder + '.txt','w')
    
    for name in res:
        print name
        a.write(name +  '\n')
a.close()   
    
        

