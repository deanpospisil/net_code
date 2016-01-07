# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:28:57 2016

@author: dean
"""

# analysis

#takes a set of points in apc plane and makes prediction based on different receptive fields
def apc_models( apc_points = [ [1], [3.14]], cur_means= [1 0.5], cur_sd= [1, 0.5], or_means= [1 0.5], or_sd= [1, 0.5]):
    #the parameters of the shapes

    mat = l.loadmat('PC2001370Params.mat')
    s = mat['orcurv'][0]
    
    #adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321] 
    a = np.hstack((range(14), range(18,318)))
    a = np.hstack((a, range(322, 370)))
    s = s[a]
        
    
    nStim = np.size(s,0) 

    angularPosition = []
    curvature = []
    paramLens = []
    
    for shapeInd in range(nStim):
        angularPosition.append(s[shapeInd][:, 0])
        curvature.append(s[shapeInd][:, 1])
        paramLens.append(np.size(s[shapeInd],0))
        
    angularPosition = np.array(list(itertools.chain.from_iterable(angularPosition)))
    angularPosition.shape = (np.size(angularPosition),1)
    
    curvature = np.array(list(itertools.chain.from_iterable(curvature)))
    curvature.shape = (np.size(curvature),1)
    
    #variable section length striding
    inds = np.empty((2,np.size(paramLens)),dtype = np.intp)
    inds[1,:] = np.cumsum(np.array(paramLens), dtype = np.intp) #ending index
    inds[0,:] = np.concatenate(([0,], inds[1,:-1])) #beginning index
    
    
    
    
#    #the Nonlin fit model for Pasupathy V4 Neurons
#    mat = l.loadmat('V4_370PC2001_LSQnonlin.mat')
#    f = np.array(mat['fI'][0])[0]
#    # orientation, curvature, orientation SD , curvature SD , correlation
#    
#    #use these to generate parameters for brute force model
#    maxAngSD = np.percentile(f[:,2], 100 - perc)
#    minAngSD = np.percentile(f[:,2], perc)
#    maxCurSD = np.percentile(f[:,3], 100 - perc)
#    minCurSD = np.percentile(f[:,3], perc)
    
    maxAngSD = np.deg2rad(171)
    minAngSD = np.deg2rad(23)
    maxCurSD = 0.98
    minCurSD = 0.09
    
    #make this into a pyramid based on d-prime
    orMeans = np.linspace(0, 2*pi-2*pi/nMeans, nMeans) 
    orSDs = np.logspace(np.log10(minAngSD),  np.log10(maxAngSD),  nSD)
    curvMeans = np.linspace(-0.5,1,nMeans)
    curvSDs = np.logspace(np.log10(minCurSD),  np.log10(maxCurSD),  nSD)
    modelParams = cartesian([orMeans,curvMeans,orSDs,curvSDs])
    nModels = np.size( modelParams, 0)
    
    a = st.vonmises.pdf(angularPosition, kappa = modelParams[:,2]**-1 , loc =  modelParams[:,0]) # 
    b = st.norm.pdf(curvature, modelParams[:,1],  modelParams[:,3])
    temp = a * b

    models = np.empty(( 362, nModels ))
    
    for shapeInd in range(nStim):
        models[ shapeInd, : ] = np.max( temp[ inds[ 0, shapeInd ] : inds[ 1 , shapeInd ] , : ] ,  axis = 0 )
    
    models = models - np.mean(models,axis = 0)
    magnitude = np.linalg.norm( models, axis = 0)
    magnitude.shape=(1,nModels)
    models = models / magnitude
    del a,b, temp
    return models, modelParams