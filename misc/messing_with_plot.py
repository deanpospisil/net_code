# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:54:18 2016

@author: deanpospisil
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt 
import numpy as np
from itertools import product
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import matplotlib
from matplotlib.ticker import FuncFormatter
import pickle as pk
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.ticker as mtick;
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp

def naked_plot(axes):
    for ax in  axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
def fit_axis(ax, x, y, x_ax=True, y_ax=True, unity_ax=True):
    maxest = max([max(x), max(y)])
    minest = min([min(x), min(y)])
    ax.plot([minest,maxest],[minest,maxest], lw=0.1, color='k');
    if min(y)<0:
        ax.plot([min(x),max(x)],[0,0], lw=.1, color='k');
    if min(x)<0:
        ax.plot([0,0],[min(y),max(y)], lw=.1, color='k');

def small_mult_scatter_w_marg(x, y):
    m = y.shape[1]+1
    n = x.shape[1]+1
    left_bottom = m*n-n
    y_hist_pos = list(range(0, m*n, n))[:-1]
    x_hist_pos = list(range(left_bottom+1, m*n))
    
    scatter_inds = list(set(range(m*n)) - (set(x_hist_pos) | set(y_hist_pos) | set([left_bottom,])))
    cart_inds = list(product(range(m-1), range(n-1)))
    
    gs = gridspec.GridSpec(m, n,                    
                            width_ratios=[1,]+[8,]*(n-1),
                            height_ratios=[8,]*(m-1)+[1,]
                           )
    
    plt.figure(figsize=(n,m))
    fontsize=6
    y_hists = []
    for y_var, pos in zip(y.T, y_hist_pos):
        _=plt.subplot(gs[pos])
        n =_.hist(y_var, orientation='horizontal',histtype='step',
                  align='mid', bins='auto',lw=0.5)[0]
        _.plot([0,0],[np.min(y_var),np.max(y_var)], color='k',lw=0.5)
        _.set_xlim(-0.1,max(n)+max(n)*.15)
        numbers = [np.min(y_var),]
    
        for number in numbers:
            _.text(0, number, np.round(number,2), ha='right', va='top',fontsize=fontsize)
        _.text(0, np.max(y_var), np.round(np.max(y_var), 1), ha='right', va='bottom',fontsize=fontsize)
        _.set_ylabel(str(pos), rotation='horizontal', 
                     labelpad=fontsize*3, fontsize=fontsize)
    
        y_hists.append(_)
    x_hists = []
    
    for x_var, pos in zip(x.T, x_hist_pos):
        _ = plt.subplot(gs[pos])
        n = _.hist(x_var,histtype='step',align='mid',lw=0.5)[0]
        _.plot([np.min(x_var),np.max(x_var)],[0,0], color='k',lw=0.9)
        _.set_ylim(0,max(n)+max(n)*.15)
        _.text(np.max(x_var), 0, np.round(np.max(x_var),1),  ha='right',va='top', fontsize=fontsize)
        _.text(np.min(x_var), 0, np.round(np.min(x_var),1),  ha='left',va='top', fontsize=fontsize)
        
        _.set_xlabel(str(pos), rotation='horizontal', 
                     labelpad=fontsize, fontsize=fontsize)
        x_hists.append(_)
    
    scatters = []    
    for (y_ind, x_ind), pos in zip(cart_inds, scatter_inds):
        _ = plt.subplot(gs[pos], sharex= x_hists[x_ind], sharey=y_hists[y_ind])
        _.scatter(x[:, x_ind], y[:, y_ind], s=0.1)
        fit_axis(_, x[:, x_ind], y[:, y_ind])
        scatters.append(_)
        
    naked_plot(x_hists + y_hists + scatters)

    return scatters, x_hists, y_hists

v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()

apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                  dmod.chunk({}), 
                                  fit_over_dims=None, 
                                  prov_commit=False)

v4_resp_apc_pd = v4_resp_apc[:,apc_fit_v4.argsort().values].to_pandas()

best_mods_pd = dmod[:, apc_fit_v4[apc_fit_v4.argsort().values]
                  .squeeze().coords['unit'].models.values]


fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
apc_fit_v4 = apc_fit_v4**2
fit_best_mods_pd = []
for resp, mod in zip(best_mods_pd.values.T, v4_resp_apc_pd.values.T):
    mod = np.expand_dims(mod, 1)
    resp = np.expand_dims(resp, 1)
    fit_best_mods_pd.append(np.dot(mod, np.linalg.lstsq(mod, resp)[0]))
fit_best_mods_pd = np.array(fit_best_mods_pd).squeeze().T
fit_best_mods_pd = pd.DataFrame(fit_best_mods_pd, columns=np.round(np.sort(apc_fit_v4.values),3))


