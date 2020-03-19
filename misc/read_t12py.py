# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:06:37 2017

@author: deanpospisil
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
cells = os.listdir(top_dir+ 'data/responses/apc_orig')
text_file = open("filename.dat", "r")
lines = text_file.readlines()