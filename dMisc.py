# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:06:16 2015

@author: dean
"""
import numpy as np
import os
pi = np.pi

def circPoints(center, radius, theta):
    circ = center[:,None] + radius * np.array([np.cos(theta), np.sin(theta)])
    return circ 

def nonNormalizedVonMises(x, mean, spread):
    # zeros is uniform, mean starts out centered at 0
    #x=np.array(x)
    
    return np.exp((spread**-1)*np.cos(x-mean))#dont need to normalize this as my measure of fit is correlation
    
def myGuassian(x,mu,sig):
    #x=np.array(x)
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def numericalWrappedGaussian(x,mean,std, stdWrapLim):
    #sums up the 2*pi*value*k (k all integers), up to a limit
    #go to the right, then to the left
    # running tests if I go more than 39 stds returned value is 0,just before that is 10**-314
    nIter=np.ceil(stdWrapLim/(2*pi/std))
    k=0
    y = myGuassian( x, mean, std)
    while k<nIter:
        
        #add up unwrapping left and right simultaneously
        k+=1
        y += myGuassian( x - 2*pi*k, mean, std) + myGuassian( x + 2*pi*k, mean, std)
        
    return y

def sectStrideInds(stackSize, length):
    #returns a list of indices that will cut up an array into even stacks, except for
    # the last one if stackSize does not evenly fit into length
    remainder = length % stackSize
    a = np.arange( 0, length, stackSize)
    b = np.append(np.arange( stackSize, length, stackSize ), length)
    stackInd = np.intp(np.vstack((a,b))).T
    
    return stackInd, remainder

def ifNoDirMakeDir(dirname):
    dirlist = dirname.split('/')
    dname = ''
    for d in dirlist:
        dname = dname + '/' +d
        if not os.path.exists(dname):
            os.mkdir(dname)

def provenance_commit(cwd):
    
    from os import system
    from git import Repo
    from datetime import datetime
    
    repo = Repo( cwd )
    assert not repo.bare
    
    #making a message. of when the commit was made
    time_str = str(datetime.now())
    time_str = 'provenance commit ' + time_str
    
    commit_message = "git commit -a -m  %r." %time_str
    system(commit_message)
    sha = repo.head.commit.hexsha
    
    return sha


##lets just look at these gaussians
#import matplotlib.pyplot as plt
#x = np.linspace(-pi,pi, 100)
#m = -1
#std = 0.9
#
#y = numericalWrappedGaussian(x,m,std,20)
#
#plt.plot(x,y)
#