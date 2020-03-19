# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:10:49 2016

@author: deanpospisil
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

a = ax.text(0, 0, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
        , clip_on=True)
