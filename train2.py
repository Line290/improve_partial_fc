import sys
import numpy as np
import mxnet as mx
import time
import cPickle
# import custom_layers
import logging
import os
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "32"
import skcuda.cublasxt as cublasxt
import math
import os
import scipy

from data_iter import DataIter
from mixmodule3 import create_net, mixModule
sys.path.append('/train/execute/')
from DataIter import PersonReID_Proxy_Batch_Plate_Mxnet_Iter2_NSoftmax_new

# ctx = [mx.gpu(i) for i in range(3)]
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.info('mxnet version %s', mx.__version__)
ctx = [mx.gpu(0)]
handle = cublasxt.cublasXtCreate()
# mode = cublasxt.cublasXtGetPinningMemMode(handle)
cublasxt.cublasXtSetPinningMemMode(handle, 1)
cublasxt.cublasXtSetCpuRatio(handle, 0, 0, 0.9)
nbDevices = len(ctx)
deviceId = np.array(range(nbDevices), np.int32)
cublasxt.cublasXtDeviceSelect(handle, nbDevices, deviceId)

num_epoch = 1000000
batch_size = 64*nbDevices
show_period = 1000

assert(batch_size%nbDevices==0)
bsz_per_device = batch_size / nbDevices

logging.info('batch_size per device: %d', bsz_per_device)
featdim = 512
total_proxy_num = 285000
data_shape = (batch_size, 3, 240, 120)
proxy_yM_shape = (batch_size, 1)
# proxy_Z_shape = (featdim, total_proxy_num)
proxy_Z_fn = '../proxy_Z.params'
proxy_Z_fn_save = 'proxy_Z.params'
proxy_Z = mx.nd.random.uniform(low=-0.5, high=0.5, 
                                  shape=(total_proxy_num, featdim), 
                                  dtype='float32', 
                                  ctx=mx.cpu(0))
print proxy_Z.shape

if os.path.exists(proxy_Z_fn):
    proxy_Z = mx.nd.load(proxy_Z_fn)[0]
    assert(total_proxy_num==proxy_Z.shape[0])
    logging.info('load proxy_Z from : %s', proxy_Z_fn)
dlr = 1050000/batch_size
radius = 32
hardratio = 10**-5
lr_start = 0.06
lr_min = 10**-5
lr_reduce = 0.96 #0.99
lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
lr_stepnum = np.int(np.ceil(lr_stepnum))
dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
logging.info('lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d',lr_start, lr_min, lr_reduce, lr_stepnum)
#     print dlr_steps
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)

# param_prefix = 'MDL_PARAM/params5_proxy_nca-8wmargin_20180724_dim_512_bn/person_reid-back'
save_prefix = './'
# load_paramidx = 0 #None

# DataBatch
# data_path_prefix = '/train/trainset/list_clean_28w_20180803'
# data_path_prefix = '/train/execute/improve_partial_fc/dataset/list_clean_28w_20180803'
# data_iter = DataIter(prefix = data_path_prefix, image_shapes = data_shape, data_nthreads = 4)

proxy_batch = 1049287
proxy_num = 285000

datafn_list = ['/train/execute/listFolder/trainlist_reid/list_all_20180723.list']
data_iter = PersonReID_Proxy_Batch_Plate_Mxnet_Iter2_NSoftmax_new(['data'], [data_shape], 
                                                               ['proxy_yM'], [proxy_yM_shape], 
                                                               datafn_list, 
                                                               total_proxy_num, 
                                                               featdim, 
                                                               proxy_batch, 
                                                               proxy_num, 1)
# simple DataBatch test
# data_batch = mx.random.normal(0, 1.0, shape=data_shape)
# data_label = mx.nd.array(range(64), dtype='int32')
# data_test = mx.io.DataBatch(data=[data_batch], label=[data_label])
param_prefix = 'MDL_PARAM/params5_proxy_nca-8wmargin_20180724_dim_512_bn/person_reid-back'
load_paramidx = 2 #None

data = mx.sym.Variable('data')
part_net = create_net(data, radius)
mxmod = mixModule(
                symbol        =     part_net, 
                context       =     ctx, 
                handle        =     handle,
                hardratio     =     hardratio,
                data_shape    =     data_shape, 
                proxy_Z       =     proxy_Z, 
                K             =     999, 
                param_prefix  =     param_prefix, 
                load_paramidx =     load_paramidx)

# mxmod.init_params(mx.init.Xavier())
mxmod.init_optimizer(optimizer="sgd", 
                   optimizer_params={
                       "learning_rate": lr_start,
                       'lr_scheduler':lr_scheduler,
                       'clip_gradient':None,
#                        "wd": 0.0005,
#                            "beta1": beta1,
                   })


for epoch in range(num_epoch):
# for epoch in range(1):
    logging.info('Epoch [%d] start...',epoch)
    data_iter.do_reset()
    start = time.time()
#     data_test = data_iter.next()
    for batch_iter in range(dlr):
#     for batch_iter in range(1):
#         print type(data_iter.next())
#         print data_iter.next().data[0].shape, data_iter.next().label[0].shape
        mxmod.update(data_iter.next())
#         mxmod.update(data_test)
        if batch_iter % 5 == 0:
            logging.info('Iter [%d]   speed %.2f samples/sec   loss = %.2f   ang = %.2f', batch_iter, batch_size*5/(time.time()-start), mxmod.get_loss().asnumpy(), mxmod.get_ang(radius).asnumpy())
            start = time.time()
        if batch_iter % 100 == 0:
            mxmod.save_checkpoint(save_prefix, epoch)
            mxmod.save_proxy_Z(proxy_Z_fn = proxy_Z_fn_save)