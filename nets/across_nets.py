# this is for getting responses across a bunch of nets for a set of stimuli.
#name the directory          

import caffe_net_response as cf                                                   
ann_dir = '/home/dean/caffe/models/bvlc_reference_caffenet/'
ann_fn = 'caffenet_train_iter_10000'

#choose a library of images
baseImageList = ['PC370', 'formlet']
base_image_nm = baseImageList[0]

#get stimuli list
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=range(370), 
                                                             blur=None, 
                                                             scale =None,  
                                                             x=(-1, 1, 101), 
                                                             y=None, 
                                                             rotation = None) 
#you will need to write a whole bunch of individual files
da = cf.get_net_resp(base_image_nm, ann_dir, ann_fn, stim_trans_cart_dict,
                 require_provenance=True)

resp_name = cf.get_net_resp_name(stim_trans_dict, ann_fn, base_image_nm)

#then you will need to concatenate them into one dataset
def read_netcdfs(files, dim):
    # glob expands paths with * to a list of files, like the unix shell
    paths = sorted(glob(files))
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(dataset, dim)
    return combined