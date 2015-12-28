# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:47:37 2015

@author: dean
"""

#sys.path.append('/Users/dean/Desktop/Deans_Modules/')
import numpy as np
import scipy as sc
import dImgProcess as imp
import os
import pickle
import dMisc as misc
############ 

# give it a list of folders and it will get their parameters from insid and generate 
#images from it
baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseStim = baseImageList[0] 


transNames = ['shape', 'x', 'y','scale', 'rot' ]
#these are the transformations you will make
stochastic = False


stimDir = '/Users/dean/Desktop/shapenet/stim/'
baseStimDir = stimDir + 'basestim/'
transStimDir = stimDir + 'trans_stim/'


genStim = baseStim + '_stoch'
os.chdir( transStimDir + genStim)
files = os.listdir(os.getcwd())

transDict = {}
for fnm in files:
    if baseStim not in fnm:
        os.remove(fnm)
    else:
        transDict = pickle.load( open( fnm, "rb" ) )





stackSize = 200
defaultStackSize =stackSize

#these will be the indices for pulling out stacks
#remainder tells you the length of the last bit if there was a remainder
nImgs = np.size(transDict['shape'])
stackInd, remainder = misc.sectStrideInds(stackSize, nImgs)
nPass = np.size(stackInd,0)


# preload image directory
imgIDs = np.linspace(np.min(transDict['shape']), np.max(transDict['shape']), np.max(transDict['shape'])+1 )
imgInd = 0

fnm = baseStimDir+ baseStim + '/' + '0'+ '.npy'
firstImg = np.load(fnm)
shapeSet = np.zeros((np.size( imgIDs ), firstImg.shape[0], firstImg.shape[1]))

for imgName in imgIDs :
    fnm = baseStimDir+ baseStim + '/' + str(int(imgName)) + '.npy'
    shapeSet[ imgInd, :, : ]  = np.load(fnm)
    imgInd+=1



paramInd = -1
stack = np.empty( ( defaultStackSize, 3, shapeSet.shape[1], shapeSet.shape[2]) )
stack[:] = np.NAN
stackSize = defaultStackSize
  
#nPass=1
#this is each time you pass a stack of images through the net
for passInd in range(nPass):
        
    beg = stackInd[passInd,0 ]
    fin = stackInd[passInd,1 ]
    
    #load the ind Dict
    sectDict = {}
    for key, i in zip(transNames, range(len(transNames))):
        if key in transDict:        
            sectDict[key] = transDict[key][beg:fin]  
    
    #fill the shape set with the initial shapes
    trans_stack = shapeSet[ np.intp( sectDict['shape']) , :, : ]    
    
    #transform the stack according to the stack of params in secDict
    trans_stack = imp.imgStackTransform( sectDict, trans_stack )
    stack = np.tile(trans_stack, (3,1,1,1))


    for saveInd in range(np.size( trans_stack, 0 )):
        paramInd+=1
        shapeId = np.int( sectDict['shape'][saveInd] )
        fileName = str(paramInd) + '_' + str(shapeId) 
        sc.misc.imsave( transStimDir + genStim + '/' + fileName + '.bmp', stack[:, saveInd, :,:])


    print(passInd/np.double(nPass))
    
#
#
##randomly divide the images up into the three folders.
#perc = np.arange(0,nImgs)/np.double(nImgs-1)
#TrainValOrTest = np.zeros(np.size(perc), dtype = np.int)
#TrainValOrTest[perc<0.5] = 0
#TrainValOrTest[perc>0.5] = 1
#TrainValOrTest[perc>0.75] = 2
#np.random.shuffle(TrainValOrTest)
#
#trainvaltestFolderNames[TrainValOrTest[paramInd]]
#
##writing down image ID's
#os.chdir(saveImgDir)
#
##do this for each folder
#for folder in trainvaltestFolderNames:
#    
#    print folder
#    
#    #look at each file
#    files = os.listdir( saveImgDir + folder )
#    #based on its name write down its location and its ID
#    res = [(k + ' ' + k.split('_')[1].split('.')[0]) for k in files if 'bmp' in k]
#    
#    a = open( folder + '.txt','w')
#    
#    for name in res:
#        print name
#        a.write(name +  '\n')
#a.close()   
#    
#        
#
