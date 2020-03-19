# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:37:52 2016

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:09:52 2016

@author: deanpospisil
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:45:44 2016

@author: deanpospisil
"""

import sys, os
import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

top_dir = os.getcwd().split('net_code')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'net_code/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')
sys.path.append(top_dir + 'nets')
plt.close('all')

import xarray as xr
def polar2cart(r, theta, center):

    x = r  * np.cos(theta) + center[0]
    y = r  * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center, final_radius, initial_radius = None, phase_width = 3000):

    if initial_radius is None:
        initial_radius = 0

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim ==3:
        polar_img = img[Ycart,Xcart,:]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width,3))
    else:
        polar_img = img[Ycart,Xcart]
        polar_img = np.reshape(polar_img,(final_radius-initial_radius,phase_width))

    return polar_img
    
def my_cor(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    r = np.dot(a, b)
    return r
def vis_square(data, padsize=0, padval=0):

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data, interpolation='nearest', cmap = cm.hot, vmin=0, vmax=1)
    plt.colorbar()

    plt.tight_layout()
    return data


def get2dCfIndex(xsamps, ysamps,fs):
    fx, fy = np.meshgrid(np.fft.fftfreq(int(xsamps),1./fs),
                         np.fft.fftfreq(int(ysamps),1./fs) )
    c = fx + 1j * fy
    return c


if 'afile' not in locals():
    with open(top_dir + 'nets/netwts.p', 'rb') as f:

        try:
            afile = pickle.load(f, encoding='latin1')
        except:
            afile = pickle.load(f)
def coef_var(a):
    mu = a.mean()
    sig = a.std()
    return 1./(((sig/mu)**2)+1)
    
def sinusoid_weights_test(orientation_of_inputs, weights_on_outputs, freq=2):
    lyr_2_prd_df = xr.DataArray([np.cos(freq*orientation_of_inputs.values), 
                                 np.sin(freq*orientation_of_inputs.values)], 
                                dims=['p','l1'])
    lyr_2_prd_df_nrm = lyr_2_prd_df / (lyr_2_prd_df**2).sum('l1')**0.5
    
    fits = (lyr_2_prd_df_nrm*lyr_2_wts_df).sum('l1').squeeze()
    lyr_2_wts_df_hat = (fits * lyr_2_prd_df_nrm).sum('p')

    lyr_2_wts_df_hat_nrm = lyr_2_wts_df_hat / (lyr_2_wts_df_hat**2).sum(['l1','r','c'])**0.5
    lyr_2_wts_df_nrm = lyr_2_wts_df / (lyr_2_wts_df**2).sum(['l1','r','c'])**0.5
    cor = (lyr_2_wts_df_hat_nrm * lyr_2_wts_df_nrm).sum(['l1','r','c'])
    return cor
    
layer = 0
sample_rate_mult = 10
ims = afile[layer][1]

lyr_1_grp_sze = 48
lyr_2_grp_sze = 128

first_layer_weights = np.array([im for im in ims])
first_layer_weights_grey_scale = np.sum(ims, 1)[:lyr_1_grp_sze, ...]
first_layer_weights_grey_scale -= np.mean(first_layer_weights_grey_scale, axis =(1,2), keepdims=True)
upsampled_fft_amplitude = np.abs(np.fft.fft2(first_layer_weights_grey_scale, 
                        s=np.array(np.shape(first_layer_weights_grey_scale)[1:])*sample_rate_mult))

polar = [img2polar(np.fft.fftshift(a_filter), [55,55], 55, phase_width=360)
                    for a_filter in upsampled_fft_amplitude]
    
polar_amp_kurtosis = np.array([coef_var(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_deg = np.array([np.argmax(polar_filter.sum(0)) for polar_filter in polar])
prfrd_ori_rad = np.deg2rad(prfrd_ori_deg)
prfrd_ori_rad_wrp = prfrd_ori_rad%np.pi

power_concentration = upsampled_fft_amplitude.max((1,2)) / upsampled_fft_amplitude.sum((1,2))
top_pwr_cncntrtn_ind = polar_amp_kurtosis<np.percentile(polar_amp_kurtosis, 80)

lyr_2_wts = afile[layer+1][1][:lyr_2_grp_sze, ...]
lyr_2_wts = np.swapaxes(lyr_2_wts, 1, 3)
lyr_2_wts_df = xr.DataArray(lyr_2_wts, dims=['l2', 'r', 'c', 'l1'])
prfrd_ori_rad_wrp_df = xr.DataArray(prfrd_ori_rad_wrp, dims=['l1'])

all_cors = []
all_null_cors = []
freqs = np.linspace(0, 10, 100)
null_prfrd_ori_rad_wrp = prfrd_ori_rad_wrp_df.values
n_resamples = 1
for freq in freqs:
    print(freq/max(freqs))
    cor = sinusoid_weights_test(prfrd_ori_rad_wrp_df, lyr_2_wts_df, freq=freq)
    all_cors.append(cor)
    for ind in range(n_resamples):
        null_prfrd_ori_rad_wrp = np.random.permutation(null_prfrd_ori_rad_wrp)
        null_prfrd_ori_rad_wrp_df = xr.DataArray(null_prfrd_ori_rad_wrp, dims=['l1'])
        null_cor = sinusoid_weights_test(null_prfrd_ori_rad_wrp_df, lyr_2_wts_df, freq=freq)
    all_null_cors.append(np.array(null_cor).flatten())
  #%%  
plt.figure()
all_cors = np.array(all_cors)
#plt.plot(freqs, np.max(all_cors,1))
plt.plot(freqs, np.percentile(all_cors, 75, axis=1))
plt.plot(freqs, np.median(all_cors,1))
plt.plot(freqs, np.percentile(all_cors, 25, axis=1))
#plt.plot(freqs, np.min(all_cors,1))
plt.plot(freqs, np.percentile(all_null_cors, 95, axis=1), alpha=0.8)
for freq, a_cor in zip(freqs, all_cors):
    plt.scatter(np.array([freq,]*128), a_cor, s=0.1, c='b', marker='.', alpha=0.5)

for freq, a_cor in zip(freqs, all_null_cors):
    plt.scatter(np.array([freq,]*128), a_cor, s=0.1, c='b', marker='.', alpha=0.5)


plt.ylim([0, 1])
plt.ylabel('Correlation')
plt.xlabel('Frequency cycles/orientation')
plt.legend(['max', '75th percentile' , 'median','25th percentile', 'null 95th percentile'])
plt.legend(['75th percentile' , 'median','25th percentile'])
fn = '/Users/deanpospisil/Desktop/'
#plt.savefig(fn + 'cor_vs_freq_with__dots.eps', )
plt.savefig(fn + 'cor_vs_freq_with__dots.png', )

'''
#now do null distribution
alt_cor = sinusoid_weights_test(prfrd_ori_rad_wrp_df, lyr_2_wts_df, freq=2)

n_resamples = 100
null_prfrd_ori_rad_wrp = prfrd_ori_rad_wrp
all_null_cors = []
for ind in range(n_resamples):  
    null_prfrd_ori_rad_wrp = np.random.permutation(null_prfrd_ori_rad_wrp)
    null_prfrd_ori_rad_wrp_df = xr.DataArray(null_prfrd_ori_rad_wrp, dims=['l1'])
    cor = sinusoid_weights_test(null_prfrd_ori_rad_wrp_df, lyr_2_wts_df, freq=freq)
    all_null_cors.append(cor)

plt.hist(np.array(all_null_cors).ravel(), range=[0,1],  histtype='step', bins=100, normed=True, cumulative=True)
plt.hist(np.array(alt_cor).ravel(), range=[0,1], histtype='step', bins=100, normed=True, cumulative=True)
plt.legend(['Shuffled Orientations', 'Original Orientations'], loc=4)
plt.xlabel('Correlation')
plt.ylim(0,1.1)
plt.grid()
plt.ylabel('Fraction < Correlation')
plt.savefig(fn + 'shuffled_vs_original_orientations_cumhist.eps', )

plt.figure()
plt.hist(np.array(all_null_cors).ravel(), range=[0,1],bins=100,  histtype='step', normed=True)
plt.hist(np.array(alt_cor).ravel(), range=[0,1],bins=30, histtype='step', normed=True)
plt.legend(['Shuffled Orientations', 'Original Orientations'], loc=1)
plt.xlabel('Correlation')
plt.ylabel('Density')
plt.savefig(fn + 'shuffled_vs_original_orientations_hist.eps', )


unrav_over_last = (np.product(np.shape(ims_2)[:-1]), np.shape(ims_2)[-1])
b = np.reshape(ims_2, unrav_over_last)

ors = np.squeeze(ors)
sorsi = np.argsort(ors)
ors = ors[ sorsi]
b = b[:, sorsi]

freq=2
freq=2
predictor = np.array([np.cos(freq*ors), np.sin(freq*ors)]).T
predictor = predictor / np.sqrt(np.sum(predictor**2, axis=0, keepdims=True))

x, res, ran, s = np.linalg.lstsq(predictor, b.T)
per_var = res/np.sum(b**2, axis=1)
res = res.reshape(ims_2.shape[:-1])
per_var_kern = np.sum(res, axis=(1,2)) / np.sum(ims_2**2, axis=(1,2,3))

cor = np.sqrt(1-per_var)
recon = np.dot(predictor, x).T

plt.subplot(211)
plt.stem(np.sqrt(1-per_var_kern))
plt.title('2nd layer fits (fitting only '+ str(ims_2.shape[-1]) +
                ' top oriented 1st layer kernels)')

plt.ylabel('r')
plt.xlabel('Second Layer Kernel')
plt.ylim(0,1)
plt.tight_layout()


plt.subplot(212)
bf = np.argmax(cor)
loc = np.unravel_index(bf, ims_2.shape[:-1]) 
plt.plot(np.rad2deg(ors), recon[bf,:])
plt.scatter(np.rad2deg(ors), b[bf,:])
plt.ylabel('kernel weight')
plt.xlabel('Orientation (degrees)')
plt.xlim(0,180)
plt.title('best fit kernel pixel r = ' +str(np.round(cor[bf], decimals=2)))
plt.tight_layout()

#plt.subplot(313)
#recon_orig = np.reshape(recon, ims_2.shape[:])
#recon_orig = recon_orig[31,...].reshape(25, 38)
#
#b_orig = np.reshape(b, ims_2.shape[:])
#b_orig = b_orig[31,...].reshape(25, 38)
#
#for ind in range(1):
#    plt.plot(np.rad2deg(ors), recon_orig[ind,:])
#    plt.scatter(np.rad2deg(ors), b_orig[ind,:])
    
    

print(loc)
print(cor[bf])
print(my_cor(recon[bf,:], b[bf,:]))

cor_im = np.reshape(cor, ims_2.shape[:-1])
plt.figure()
data = vis_square(cor_im, padsize=2, padval=0)
plt.xticks([])
plt.yticks([])
plt.title('Plaid-Preference Model r Map')

x, res, ran, s = map(np.array, zip(*[np.linalg.lstsq(predictor, b.T) 
                for ind in range(10)]))


plt.figure()
data = ims
n = int(np.ceil(np.sqrt(data.shape[0])))
data = (data - data.min()) / (data.max() - data.min())

#for ind in range(len(data)):
#    plt.subplot(10, 10,ind+1)
#    _ = (data[ind] - data[ind].min()) / (data[ind].max() - data[ind].min())
#    plt.imshow(_, interpolation='None',cmap=cm.Greys_r )
#    plt.gca().set_xticks([])
#    plt.gca().set_yticks([])
#    plt.title(str(ind) + ': ' + str(np.round(np.rad2deg(ors[ind]))))
#plt.tight_layout()

ors = np.squeeze(ors)[:48]
predictor = np.array([ np.cos(2*ors), np.sin(2*ors)])

predictor = predictor / np.sum(predictor**2, 0, keepdims=True)
ims_2 = afile[layer+1][1]
resper = []
xs=[]

for ind in range(128):
    b = ims_2[ind,:, 2, 2]
    x,res,ran,s = np.linalg.lstsq(predictor.T, b)
    xs.append(x)
    resper.append(res/(np.linalg.norm(b)**2))


resper = np.sqrt(1-np.array(resper))
plt.figure()
plt.subplot(211)
plt.plot(resper)
plt.ylabel('r')
plt.xlabel('Second Layer Kernel Ind')
plt.tight_layout()

bf = np.argmax(resper)
sorsi = np.argsort(ors)
plt.subplot(212)



plt.scatter(np.rad2deg(ors[sorsi]), ims_2[bf,sorsi,2,2])
plt.plot(np.rad2deg(ors[sorsi]), (np.dot(np.expand_dims(xs[bf],0), predictor)).T[sorsi], color='r')
plt.ylabel('kernel weight')
plt.xlabel('Orientation (degrees)')
plt.xlim(0,180)
plt.title('best fit kernel: ' + str(bf))
plt.tight_layout()
'''
#mst_orintd_1st_lyr_wts = first_layer_weights_grey_scale[top_pwr_cncntrtn_ind, ...]

#for ind in  polar_amp_kurtosis.argsort()[0:2]:
#    plt.figure()
#    plt.imshow(first_layer_weights_grey_scale[ind],
#                interpolation='nearest', cmap=plt.cm.Greys_r)
#    plt.figure()
#    plt.imshow(np.fft.fftshift(upsampled_fft_amplitude[ind]),
#                interpolation='nearest', cmap=plt.cm.Greys_r)
#    plt.figure()
#    plt.imshow(polar[ind], interpolation='nearest', cmap=plt.cm.Greys_r)
#    plt.figure()
#    plt.plot(polar[ind].sum(0))