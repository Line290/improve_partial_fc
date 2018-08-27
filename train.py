import sys
import numpy as np
import mxnet as mx
import time
import cPickle
# import custom_layers
import logging

import skcuda.cublasxt as cublasxt
import math
import os
import scipy

from data_iter import DataIter
from mixmodule import create_net, mixModule

print 'mxnet version' + mx.__version__
# ctx = [mx.gpu(i) for i in range(3)]
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
print 'batch_size per device:', bsz_per_device

# featdim = 128
featdim = 512
total_proxy_num = 285000
data_shape = (batch_size, 3, 240, 120)
# proxy_Z_shape = (featdim, total_proxy_num)
proxy_Z_fn = './proxy_Z.npy'
proxy_Z = (np.random.rand(featdim, total_proxy_num)-0.5)*0.001
proxy_Z = proxy_Z.astype(np.float32)
# proxy_Z = mx.nd.random.uniform(low=-0.5, high=0.5, 
#                                   shape=(featdim, total_proxy_num), 
#                                   dtype='float32', 
#                                   ctx=mx.cpu(0))
# proxy_Z = proxy_Ztmp.astype(np.float32)

if os.path.exists(proxy_Z_fn):
    proxy_Z = np.load(proxy_Z_fn)
#     proxy_Z = tmpZ[0].asnumpy()
#     proxy_Z = tmpZ
#     proxy_Z = mx.nd.load(proxy_Z_fn)[0]
#     print proxy_num, tmpZ[0].shape[0]
    assert(total_proxy_num==proxy_Z.shape[1])
    print 'load proxy_Z from', proxy_Z_fn

dlr = 1050000/batch_size
radius = 0
hardratio = 10**-1
lr_start = 0.1
lr_min = 10**-5
lr_reduce = 0.96 #0.99
lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
lr_stepnum = np.int(np.ceil(lr_stepnum))
dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
print 'lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(lr_start, lr_min, lr_reduce, lr_stepnum)
#     print dlr_steps
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)

# param_prefix = 'MDL_PARAM/params5_proxy_nca-8wmargin_20180724_dim_512_bn/person_reid-back'
param_prefix = './'
load_paramidx = 0 #None

# DataBatch test
# data_path_prefix = '/train/trainset/list_clean_28w_20180803'
data_path_prefix = '/train/execute/improve_partial_fc/dataset/list_clean_28w_20180803'
data_iter = DataIter(prefix = data_path_prefix, image_shapes = data_shape, data_nthreads = 4)

# simple DataBatch test
# data_batch = mx.random.normal(0, 1.0, shape=data_shape)
# data_label = mx.nd.array(range(64), dtype='int32')
# data_test = mx.io.DataBatch(data=[data_batch], label=[data_label])

data = mx.sym.Variable('data')
part_net = create_net(data, radius)
mxmod = mixModule(symbol=part_net, 
                context=ctx, 
                handle=handle, 
                data_shape=data_shape, 
                proxy_Z=proxy_Z, 
                K = 999)

mxmod.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))
mxmod.init_optimizer(optimizer="adam", 
                   optimizer_params={
                       "learning_rate": lr_start,
                       'lr_scheduler':lr_scheduler,
                       'clip_gradient':None,
                       "wd": 0.0005,
#                            "beta1": beta1,
                   })
for epoch in range(num_epoch):
    print 'Epoch [%d] start...' % (epoch)
    data_iter.reset()
    start = time.time()
    for batch_iter in range(dlr):
#         print type(data_iter.next())
#         print data_iter.next().data[0].shape, data_iter.next().label[0].shape
        mxmod.update(data_iter.next())
#         mxmod.update(data_test)
        if batch_iter % 5 == 0:
            print 'Iter [%d]   speed %.2f samples/sec   loss = %.2f'%(batch_iter, batch_size/(time.time()-start), mxmod.get_loss())
            start = time.time()
    mxmod.save_proxy_Z(proxy_Z_fn = proxy_Z_fn)