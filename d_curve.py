# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:02:24 2015

@author: dean
"""

import warnings
import numpy as np
pi=np.pi

def curveDists(cShape):
    nPts=np.size(cShape)
    dists=np.ones([nPts])*100j
    for ind in range(nPts):
        dists[ind] = abs(cShape[ind] - cShape[ind-1 ])
        
    return dists
    

def curveCurvature(cShape):

    nPts=np.size(cShape)
    curvature=np.ones([nPts])*np.nan
    for ind in range(nPts):
        oldDir = (cShape[(ind-1)] ) - (cShape[(ind-2) ] )
        newDir = (cShape[ind] ) - (cShape[(ind-1) ] )
        curvature[ind] = np.angle(newDir * np.conj(oldDir))/abs(newDir)
    return curvature

def curveOrientations(cShape):
    #gets a complex unit vector leading up to each point going counterclockwise, 
    #then rotates it 90 degrees counterclockwise to point outwards
    nPts=np.size(cShape)
    orientation=np.ones([nPts])*1j
    for ind in range(nPts):
        #get the direction then rotate by 90 degrees clockwise for it to point outward, since points are in counterclockwise order
        orientation[ind] = (((cShape[(ind)] ) - (cShape[(ind-1) ] )))*(-1j)
        orientation[ind] = orientation[ind]/abs(orientation[ind])

    return orientation
    
    
def curveAngularPos(cShape):
    #get the center of mass than the unit vector pointing from here to each point
    centerOfMass=np.mean(cShape)
    angularPos=cShape-centerOfMass
    angularPos=angularPos/abs(angularPos)

    return angularPos    


def makeNaturalFormlet(nPts=1000, radius=1, nFormlets=32, meanFormDir=-pi, stdFormDir=pi/10, meanFormDist=1, stdFormDist=0.1, startSigma=0.3, endSigma=0.1, randomstate = None ):
    
    #set the seed for reproducibility    
    if randomstate == None:
        randomstate = np.random.RandomState(np.random.rand(1))

    if meanFormDist==np.nan:
        meanFormDist=radius
        stdFormDist=radius/10
        
#    angles = np.linspace(0.0, 2*pi-2*pi/nPts, nPts)
    angles = np.linspace(0.0, 2*pi, nPts)
    cShape = np.exp(angles*1j)*radius
    
    #where are the formlets centers going to be
    centers=np.ones(nPts)*1j
    for ind in range(nPts):
        #gaussian distributed with some bias towards a direction, but some jitter in distance from
        #origin and orientation
        centers[ind] = randomstate.normal(meanFormDist, stdFormDist)*np.exp(randomstate.normal(meanFormDir, stdFormDir)*1j)
        
    #what will be the scale of those formlets
    sigma = np.logspace( np.log10( startSigma ), np.log10( endSigma ), nFormlets) 
    
    # roughly the sigma to alpha ratiorandom sign of gain
    alpha = 0.10*sigma*2*( randomstate.binomial( 1, 0.5, nFormlets ) - 0.5 )
    
    #alpha = ((1.0/(-2.0*pi))*sigma)/1.1
    
    #apply formlet
    for ind in range(nFormlets):
        cShape = applyGaborFormlet(cShape, centers[ind], alpha[ind], sigma[ind])
    
    if cShape[0] is not cShape[-1]:
        cShape[-1] = cShape[0]
        
    return cShape, np.real(cShape), np.imag(cShape), sigma, alpha

def make_n_natural_formlets( **args ):
    rng = np.random.RandomState(args['randseed'])
    s= []
    #I did this with **args so later on I could easily return them
    for ind in range(args['n']):
        cShape, x, y, sigma, alpha = makeNaturalFormlet(nPts=args['nPts'], 
                                                    radius = args['radius'],
                                                    nFormlets = args['nFormlets'], 
                                                    meanFormDir = args['meanFormDir'],
                                                    stdFormDir = args['stdFormDir'],
                                                    meanFormDist = args['meanFormDist'], 
                                                    stdFormDist = args['stdFormDist'], 
                                                    startSigma = args['startSigma'], 
                                                    endSigma = args['endSigma'], 
                                                    randomstate = rng )
                        

        s.append( np.array( [x, y]).T )
        
    return s

def applyGaborFormlet(cShape, center, alpha, sigma):
   
   alphaBounds = [(1.0/(-2.0*pi))*sigma, 0.1956*sigma]

   r=np.abs(cShape-center)
   
 
   if alphaBounds[0]>alpha or alphaBounds[1]<alpha:
        print('alpha is outside of the bounds for which Jordan curves are guaranteed')
        warnings.warn('alpha is outside of the bounds for which Jordan curves are guaranteed')
    
   cShapeUnitVectors = (cShape-center)/r
   newcShape = center + cShapeUnitVectors * ( r + alpha * np.exp( (-r**2.0) / sigma**2.0  ) * np.sin(( 2.0 * pi * r) / sigma))
   
   return newcShape
   

