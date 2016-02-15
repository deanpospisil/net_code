# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:54:19 2015

@author: dean
"""
import pandas as pd
#receptive field analysis
import matplotlib.pyplot as p
import numpy as np
import pickle as pk
from glue.core import Data, DataCollection

p.close('all')
npts=21
theEnd = 75
a= np.linspace(-theEnd,theEnd,npts)
pos = np.vstack((a, np.zeros(np.shape(a)))).T



if 'resp' not in locals():
    f = open('/Users/dean/Desktop/AlexNet_APC_Analysis/resp7770vh.pickle')
    resp = pickle.load(f)
    f.close()
    resp=resp[0]

rfList = []
corrList = []
nLayers = np.size(resp)   
nUnits = np.zeros(nLayers)
nPos = np.shape(pos)[0]
for ind in range(nLayers):
    nUnits[ind] = np.shape(resp[ind])[0]
    

for layer in range(nLayers):
 
    #reshape
    foo = np.reshape( resp[layer], ( nUnits[layer], 370, nPos ))
    #glue.qglue( t = foo )
    
    #get receptive field (mean response to shapes at each position), for x and y
    rf=np.mean(foo,1)
    rfList.append(rf)
    
    #get correlation across translations
    tCor = np.zeros((nUnits[layer], nPos, nPos))
    
    for unit in range(int(nUnits[layer])):
        tCor[unit,:,:] = np.corrcoef(foo[unit,:,:].T)
    corrList.append(tCor)

data = Data(t=resp[0], label=str(1)) 
col=DataCollection([data])
for layer in range(2,9):
    col.append(Data(t=resp[0], label=str(layer)) )

pk.dump('bah.pickle',col)

#
##plot the mean rf across layers
#x = np.linspace(0, 1, 8)
#number = 8
#cmap = p.get_cmap('jet')
#colors = [cmap(i) for i in np.linspace(0, 1, number)]
#for i, color in enumerate(colors, start=0):
#    mrf=np.mean(rfList[i],0)
#    p.subplot(211)
#    p.plot(pos[:,0], mrf/np.max(mrf), color=color, label=i+1, lw=2)
#    p.xlabel('Stimulus Position (pix)')
#    p.ylabel('Normalized Mean Response')
#    p.yticks([0, 1.1])
#    p.xticks([])
#    p.xticks(pos[::2,0])
#    
#    mcor = np.nanmean(np.nanmean(corrList[i],0),0)
#    p.subplot(212)
#    p.plot(pos[:,0], mcor, color=color, label=i+1, lw=2)
#    p.xlabel('Stimulus Position (pix)')
#    p.ylabel('Mean Correlation')
#    p.yticks([0, 0.5, 1])
#    p.xticks(pos[::2,0])
#    
#p.show()  
#p.subplot(211)  
#p.legend(loc = 'center left', bbox_to_anchor=(0, 1.15),ncol=4, fancybox=True, title='Layer')
#plotInd=0
#fig=p.figure()
#for layer in range(nLayers):
#    plotInd+=1    
#    p.subplot(1, nLayers, plotInd)
#    im=p.imshow(np.nanmean(corrList[layer],0), interpolation='none',vmin=0, vmax=1, extent=[-75,75,-75,75])
##    p.xticks(pos[:,0])
#    p.yticks([])
#    p.xticks([])
#    if layer==0:
#        p.gca().set_yticks(pos[::5,0])
#    
##    plotInd+=1 
##    p.subplot(nLayers,2,plotInd)
##    mcor = np.nanmean(np.nanmean(corrList[layer],0),0)
##    p.plot(mcor-np.nanmin(mcor), np.flipud(pos[:,0]), lw=2)
##    p.axis([0, 1, -80, 80])
##    p.axis('off')
#    
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#c=fig.colorbar( im, cax=cbar_ax)
#c.set_ticks([0 ,0.5 ,1])
#
#posSlice = 37.5
#ind = np.where( pos[:,0]==posSlice)
#p.figure()
#for layer, color in enumerate(colors, start=0):
#    mCorSlice = np.squeeze(np.nanmean(corrList[layer],0)[ind,:])
#    p.plot(pos[:,0],  mCorSlice, color=color, label=i+1, lw=2)
#    p.xlabel('Stimulus Position (pix)')
#    p.ylabel('Mean Correlation')
#    p.yticks([0, 0.5, 1])
#    p.xticks(pos[::2,0])

