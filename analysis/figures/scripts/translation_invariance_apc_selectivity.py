# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:03:36 2016

@author: deanpospisil
"""

import os, sys
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir + 'xarray')
import xarray as xr
import os
import seaborn as sns
import matplotlib.pyplot as plt

top_dir = os.getcwd().split('net_code')[0] + 'net_code/'

imtype = '.pdf'

fname = 'apc_models_r_trans101_earlyiter'
fitm = xr.open_dataset(top_dir +'analysis/data/' + fname + '.nc' )
b = fitm.to_dataframe()
b.set_index(['layer_unit', 'layer'], append=True, inplace=True)



sns.boxplot(x="layer_label", y="cor", data=b)

fillb = b.fillna(0)

per = b[b>0.5].groupby('layer_label', sort=False).count()/fillb.groupby('layer_label', sort=False).count()
per.plot(kind = 'bar')
plt.ylim((0,1))
plt.title('Percent units > 0.5 Correlation')
fn =top_dir + 'analysis/figures/images/' +fname + '_p>5' + imtype 
plt.savefig(filename=fn)
open(fn, 'a').write("")
