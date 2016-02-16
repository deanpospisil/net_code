# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory

import caffe_net_response as cf
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
import os
import sys
top_dir = os.getcwd().split('net_code')[0] + 'net_code/'
sys.path.append(top_dir)

#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]

#get stimuli list
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(370),
                                                             blur=None,
                                                             scale =None,
                                                             x=(-50, 50, 101),
                                                             y=None,
                                                             rotation = None)

for train_iter in range(10000,990000,50000):
    ann_fn = 'caffenet_train_iter_'

    da = cf.get_net_resp(base_image_nm, ann_dir, ann_fn + str(train_iter), stim_trans_cart_dict,
                 require_provenance=True)
    ds = da.to_dataset(name = 'resp')
    ds.to_netcdf(top_dir + 'analysis/data/iter_' + str(train_iter) + '.nc')


resp_name = cf.get_net_resp_name(stim_trans_dict, ann_fn, base_image_nm)

'''
#then you will need to concatenate them into one dataset
def read_netcdfs(files, dim):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(dataset, dim)
    return combined
'''