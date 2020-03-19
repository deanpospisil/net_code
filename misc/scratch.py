# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:29:21 2017

@author: deanpospisil
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as  l
import os, sys
top_dir = os.getcwd().split('net_code')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')
import xarray as xr
import pandas as pd
fn = top_dir +'data/responses/v4_ti_resp.nc'
v4 = xr.open_dataset(fn)['resp'].load()
v4 = v4.transpose('unit','x', 'shapes') 
all_v4 = v4.values.ravel()
all_v4 = all_v4[~np.isnan(all_v4)]
ax = plt.subplot(211)
_ = plt.hist(all_v4,log=0, bins=100);plt.xlabel('Spk/Sec');plt.ylabel('Count')
pd.DataFrame(all_v4).describe()
a_v4 = v4[30,0,:]
ax = plt.subplot(212)
ax.stem(range(len(a_v4)), a_v4);ax.set_ylabel('Firing Rate');
ax.set_xlabel('Shape ID');ax.set_xlim(0,len(a_v4))
n = 5 #trials
s = 20 #stimuli
n_exps = 1000 #number simulations

a_v4 = a_v4.values

oa_v4 = a_v4.copy()
na_v4 = a_v4.copy()

model = a_v4.copy()
model -= model.mean()
model /= np.linalg.norm(model)

true_cor = []
cor_by_snr = []
#%%
for SNR in np.logspace(-10,1,100):
    residual = np.random.normal(loc=0, scale=np.ones(len(na_v4)), size=len(na_v4))
    residual = residual/np.linalg.norm(residual)
    residual = residual*np.linalg.norm(oa_v4)*SNR
    na_v4 = oa_v4 + residual
    if sum(na_v4<0)>0:
        na_v4 -= na_v4.min()*1.1
    na_v4 = (na_v4/np.linalg.norm(na_v4))*np.linalg.norm(oa_v4)
    true_cor.append(np.corrcoef(na_v4, model)[0,1])
    r = np.random.poisson(na_v4, size=(n_exps, n, len(na_v4))) # nx n s
    R = r.mean(1) # nx s
    R_ms = R - R.mean(1, keepdims=True)# nx s
    exp_cor = np.array(np.dot(R_ms/np.linalg.norm(R_ms, keepdims=True, axis=1), 
                 np.expand_dims(model, 1)))#nx 1
    cor_by_snr.append(exp_cor.mean())

plt.plot(cor_by_snr)
plt.plot(true_cor)
print(residual)