# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:19:31 2015

@author: dean
"""

#lets make the plots

import formlet
import random
from scipy import interpolate
import matplotlib.pyplot as plt
import pywt 


#make the initial complex circle
nPts=1000
angles = np.linspace(0.0, 2*pi-2*pi/nPts, nPts)
radius = 0.15
cShape = np.exp(angles*1j)*radius
startSigma=0.3
endSigma=0.1
nIter=32

#set parameters of formlet
centers=(ones(nIter)*1j)
for ind in range(nIter):
    centers[ind] = random.gauss(radius, radius/10)*exp(random.gauss(0.75*2*pi, pi/2)*1j)
sigma = np.logspace(log10(startSigma),log10(endSigma),nIter)    
alpha = 0.10*sigma*2*(np.random.binomial(1,0.5,nIter)-0.5)
#alpha = ((1.0/(-2.0*pi))*sigma)/1.1

#apply formlet
for ind in range(nIter):
    cShape = formlet.applyGaborFormlet(cShape, centers[ind], alpha[ind], sigma[ind])

x = real(cShape)
y = imag(cShape)



tck,u = interpolate.splprep([x,y], s=0)
unew = np.arange(0, 1.001, 0.001)
out = interpolate.splev(unew, tck)
cShape=out[0]+1j*out[1]

direction=formlet.getComplexJordanCurveOrientations(cShape)
direction=formlet.getComplexJordanCurveAngularPos(cShape)
magnitude=formlet.getComplexJordanCurveCurvature(cShape)
comb=(magnitude)*direction

subplot(211)
#points = array([real(cShape), imag(cShape)]).T
#line = plt.Polygon(points, closed=True, fill='b', edgecolor='none',fc='None')
#plt.gca().add_patch(line)
formlet.colorline(
     real(cShape), imag(cShape), z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=1, alpha=1)
plt.quiver(np.real(cShape),np.imag(cShape),np.real(comb),np.imag(comb),width=0.001,)        
plt.gca().set_xlim([-radius*2, radius*2])
plt.gca().set_ylim([-radius*2, radius*2])
plt.gca().set_aspect('equal')
import formlet
subplot(212)
plot(range(size(magnitude)), magnitude)


x=magnitude
coefs = pywt.wavedec(x, 'haar', level = 10, mode='per')
for ind in range(1,size(coefs),1):
    coefs[ind] = zeros(size(coefs[ind]))

rec=pywt.waverec(coefs, 'haar', mode='per')
plot(range(size(rec)),rec)
#plot()

formlet.colorline(
     range(size(rec)),zeros(size(rec)), z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1)