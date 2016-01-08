# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:06:16 2015

@author: dean
"""
import numpy as np
import os
pi = np.pi
from collections import OrderedDict as ordDict
import warnings



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
    
def cartesian_prod_dicts_lists( the_dict ) :
    
    #takes a dictionary and produces a dictionary of the cartesian product of the input
    if not type(the_dict) is type(ordDict()):
        warnings.warn('We were expecting an ordered dict for provenance concerns.')
        
    from sklearn.utils.extmath import cartesian
    
    stim_list = []
    stim_list = tuple([ list(the_dict[ key_name ]) for key_name in the_dict ])
        
    #cartesian has the last column change the fastest, thus is like c-indexing
    stim_cart_array = cartesian(stim_list)
    
    cart_dict = ordDict()
    #load up the vectors assosciated with keys to cart_dict
    for key_name, key_num in zip( the_dict, range( len( the_dict ) ) ):
        cart_dict[key_name] = stim_cart_array[ :, key_num]
    
    return cart_dict

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