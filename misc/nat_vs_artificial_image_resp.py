# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:45:09 2016

@author: deanpospisil
"""



import numpy as  np
import scipy.io as  l
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mtick
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
import pickle as pk
import pandas as pd

def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x>=1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x,2))

def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):
    
    for i, an_axes in enumerate(axes):
        if i==len(axes)-1:
            if yticks==None:
                an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
            else:
                an_axes.set_yticks(yticks)
            if xticks==None:
               an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
            else:
                an_axes.set_xticks(xticks)
                an_axes.xaxis.set_tick_params(length=0)
                an_axes.yaxis.set_tick_params(length=0)
                an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
            an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        else:
            an_axes.set_xticks([])
            an_axes.set_yticks([])
def stacked_hist_layers(cnn, logx=False, logy=False, xlim=None, maxlim=False, bins=100):
    layers = cnn.index.get_level_values('layer_label').unique()
    if logx:
        cnn = np.log(cnn.dropna())
    if maxlim:
        xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]
    for i, layer in enumerate(layers):
        plt.subplot(len(layers), 1, i+1)
        vals = cnn.loc[layer].dropna().values

        
        plt.hist(vals, log=logy, bins=bins, histtype='step', 
                 range=xlim, normed=False)
        
        plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color='red')
        plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
        plt.gca().yaxis.set_label_position("right")
    
    if logx:
        plt.xlabel('log')
    nice_axes(plt.gcf().axes) 
    #plt.suptitle(cnn.name)
    

#font = {'size' : 25}
#mpl.rc('font', **font)

cnn_names = [
'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-50, 48, 50)_ref_iter_0',
'APC362_scale_1_pos_(-50, 48, 50)_ref_iter_0',
]

v4_name = 'V4_362PC2001'
save_folder = top_dir + 'data/an_results/reference/'

#coef_var_v4 = pk.load(open(save_folder + 'coef_var' + v4_name, 'rb'))
#coef_var_alex = pk.load(open(save_folder +'coef_var' + cnn_name, 'rb'))
#eye_r2_v4 = pk.load(open(save_folder  + 'eye_r2_' + v4_name, 'rb'))
#eye_r2_alex = pk.load(open(save_folder  + 'eye_r2_' + cnn_name, 'rb'))
#k_alex = pk.load(open(save_folder  + 'k_' + cnn_name, 'rb'))
#k_v4 = pk.load(open(save_folder  + 'k_' + v4_name, 'rb'))
#ti_v4 = pk.load(open(save_folder  + 'ti_'+ v4_name, 'rb'))
#ti_alex = pk.load(open(save_folder  + 'ti_'+ cnn_name, 'rb'))
#tilc_alex = pk.load(open(save_folder  + 'trans_ill_cond_' + cnn_name, 'rb'))

apc_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp'].load()
nat_resp = xr.open_dataset(top_dir + 'data/responses/alex_net_nat_image_dist2000.nc')['resp'].load()

#
#
#    plt.close('all')
#    coef_var_v4 = pk.load(open(save_folder + 'coef_var' + v4_name, 'rb'), encoding='latin1')
#    coef_var_alex = pk.load(open(save_folder +'coef_var' + cnn_name, 'rb'), encoding='latin1')
#    eye_r2_v4 = pk.load(open(save_folder  + 'eye_r2_' + v4_name, 'rb'), encoding='latin1')
#    eye_r2_alex = pk.load(open(save_folder  + 'eye_r2_' + cnn_name, 'rb'), encoding='latin1')
#    k_alex = pk.load(open(save_folder  + 'k_' + cnn_name, 'rb'), encoding='latin1')
#    
#    k_v4 = pk.load(open(save_folder  + 'k_' + v4_name, 'rb'), encoding='latin1')
#    ti_v4 = pk.load(open(save_folder  + 'ti_'+ v4_name, 'rb'), encoding='latin1')
#    ti_alex = pk.load(open(save_folder  + 'ti_'+ cnn_name, 'rb'), encoding='latin1')
#    tilc_alex = pk.load(open(save_folder  + 'trans_ill_cond_' + cnn_name, 'rb'), encoding='latin1')
#    
#    apc_alex = pk.load(open(save_folder  + 'apc_'+ cnn_name, 'rb'), encoding='latin1')
#    apc_v4 = pk.load(open(save_folder  + 'apc_'+ v4_name, 'rb'), encoding='latin1')
#    
#    ti_v4['layer_label'] = 'v4'
#    ti_v4['layer_unit'] = range(len(ti_v4))
#    k_v4['layer_unit'] = range(len(k_v4))
#    
#    k_v4['layer_label'] = 'v4'
#    ti_v4.set_index(['layer_label', 'layer_unit'], inplace=True)
#    
#    apc_v4 = pd.concat( [apc_v4, coef_var_v4, eye_r2_v4, k_v4], axis=1).set_index(['layer_label', 'layer_unit'])
#    cnn = pd.concat([coef_var_alex, eye_r2_alex, k_alex, ti_alex, tilc_alex, apc_alex], axis=1)
#    
#    both=pd.concat([cnn, apc_v4, ti_v4])
#    both = both[(both.index.get_level_values('layer_label') != 'prob')]
#    
#
#    
#    plt.close('all')
#
#     