
import matplotlib.pyplot as plt
plt.rcdefaults()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# a function that takes a grid of complex numbers, a center, and a size, and returns
#a bunch of line segments in that grid showing orientation

dummy = np.linspace(0, 1, 25)
dummy = np.exp(dummy*1j*2*np.pi).reshape(5, 5)*0.15

center = 0.5
size = 1
nrow = ncol = 5
pos = np.mgrid[center-size/2.:center+size/2.:nrow*1j, center-size/2.:center+size/2.:ncol*1j]
pos = pos.reshape(2,nrow*ncol).T
pos = np.sum( pos * [1, 1j], 1).reshape(25,1)

for ind in range(128):
    print(ind)
    dummy = np.squeeze(fits.values[0,ind,:,:])
    dummy = dummy/abs(dummy)
    dummy = dummy.reshape(25,1)*0.1
    dummy = np.hstack([dummy,-dummy])
    dummy = dummy + pos
    
#    plt.close('all')
#    fig = plt.figure(figsize=(2,2))
#    ax = fig.gca()
    
    # add a line
    for oris in dummy:
        theplot=plt.subplot(11,12,ind+1)
        line = mlines.Line2D([np.real(oris[0]), np.real(oris[1])], 
                             [np.imag(oris[0]), np.imag(oris[1])], lw=1., alpha=1)
    
        theplot.add_line(line)
    
    
    plt.axis('equal')
    plt.axis('off')
    
plt.show()
