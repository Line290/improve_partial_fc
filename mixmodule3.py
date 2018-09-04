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

FEAT_DIM = 512
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True, namepre='', args=None):
  if args is None:
    weight = mx.sym.Variable(namepre+'_weight')
    bias = mx.sym.Variable(namepre+'_bias')
    gamma = mx.sym.Variable(namepre+'_gamma')
    beta = mx.sym.Variable(namepre+'_beta')
    args = {'weight':weight, 'bias':bias}
  else:
    weight = args['weight']
    bias = args['bias']
    gamma = args['gamma']
    beta = args['beta']
  
  conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, weight=weight, bias=bias, name=namepre+'_conv')
  bn = mx.symbol.BatchNorm(data=conv, gamma=gamma, beta=beta, name=namepre+'_bn')
  act = bn
  if with_act:
      act = mx.symbol.Activation(data=bn, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def stem(data, namepre='', args=None):
  if args is None:
    args = {'conv1a_3_3':None, 'conv2a_3_3':None, 'conv2b_3_3':None, 'conv3b_1_1':None, 'conv4a_3_3':None}
  conv1a_3_3, args['conv1a_3_3'] = ConvFactory(data=data, num_filter=32,
                           kernel=(3, 3), stride=(2, 2), namepre=namepre+'_conv1a_3_3', args=args['conv1a_3_3'])
  conv2a_3_3, args['conv2a_3_3'] = ConvFactory(conv1a_3_3, 32, (3, 3), namepre=namepre+'_conv2a_3_3', args=args['conv2a_3_3'])
  conv2b_3_3, args['conv2b_3_3'] = ConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1), namepre=namepre+'_conv2b_3_3', args=args['conv2b_3_3'])
  maxpool3a_3_3 = mx.symbol.Pooling(
      data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=namepre+'_maxpool3a_3_3')
  conv3b_1_1, args['conv3b_1_1'] = ConvFactory(maxpool3a_3_3, 80, (1, 1), namepre=namepre+'_conv3b_1_1', args=args['conv3b_1_1'])
  conv4a_3_3, args['conv4a_3_3'] = ConvFactory(conv3b_1_1, 192, (3, 3), namepre=namepre+'_conv4a_3_3', args=args['conv4a_3_3'])

  return conv4a_3_3, args 


def reductionA(conv4a_3_3, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv2_0':None, 'tower_conv2_1':None, 'tower_conv2_2':None, 'tower_conv3_1':None}
  maxpool5a_3_3 = mx.symbol.Pooling(
      data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=namepre+'_maxpool5a_3_3')

  tower_conv, args['tower_conv'] = ConvFactory(maxpool5a_3_3, 96, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(maxpool5a_3_3, 48, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])

  tower_conv2_0, args['tower_conv2_0'] = ConvFactory(maxpool5a_3_3, 64, (1, 1), namepre=namepre+'_tower_conv2_0', args=args['tower_conv2_0'])
  tower_conv2_1, args['tower_conv2_1'] = ConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_1', args=args['tower_conv2_1'])
  tower_conv2_2, args['tower_conv2_2'] = ConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_2', args=args['tower_conv2_2'])

  tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
      3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg', name=namepre+'_tower_pool3_0')
  tower_conv3_1, args['tower_conv3_1'] = ConvFactory(tower_pool3_0, 64, (1, 1), namepre=namepre+'_tower_conv3_1', args=args['tower_conv3_1'])
  tower_5b_out = mx.symbol.Concat(
      *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
  return tower_5b_out, args 


def reductionB(net, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv1_2':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 384, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv1_2, args['tower_conv1_2'] = ConvFactory(tower_conv1_1, 384, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv1_2', args=args['tower_conv1_2'])
  tower_pool = mx.symbol.Pooling(net, kernel=(
      3, 3), stride=(2, 2), pool_type='max', name=namepre+'_tower_pool')
  net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])

  return net, args


def reductionC(net, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv0_1':None, 'tower_conv1':None, 'tower_conv1_1':None, 'tower_conv2':None, 'tower_conv2_1':None, 'tower_conv2_2':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv0_1, args['tower_conv0_1'] = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv0_1', args=args['tower_conv0_1'])
  tower_conv1, args['tower_conv1'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv1', args=args['tower_conv1'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv2, args['tower_conv2'] = ConvFactory(net, 256, (1, 1), namepre=namepre+'_tower_conv2', args=args['tower_conv2'])
  tower_conv2_1, args['tower_conv2_1'] = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_1', args=args['tower_conv2_1'])
  tower_conv2_2, args['tower_conv2_2'] = ConvFactory(tower_conv2_1, 320, (3, 3),  stride=(2, 2), namepre=namepre+'_tower_conv2_2', args=args['tower_conv2_2'])
  tower_pool = mx.symbol.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type='max', name=namepre+'_tower_pool')
  net = mx.symbol.Concat(*[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])
  return net, args


def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv2_0':None, 'tower_conv2_1':None, 'tower_conv2_2':None, 'tower_out':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 32, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 32, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv2_0, args['tower_conv2_0'] = ConvFactory(net, 32, (1, 1), namepre=namepre+'_tower_conv2_0', args=args['tower_conv2_0'])
  tower_conv2_1, args['tower_conv2_1'] = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_1', args=args['tower_conv2_1'])
  tower_conv2_2, args['tower_conv2_2'] = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1), namepre=namepre+'_tower_conv2_2', args=args['tower_conv2_2'])
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
  tower_out, args['tower_out'] = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False, namepre=namepre+'_tower_out', args=args['tower_out'])

  net = net + scale * tower_out
  act = net
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv1_2':None, 'tower_out':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 192, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 129, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv1_2, args['tower_conv1_2'] = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1), namepre=namepre+'_tower_conv1_2', args=args['tower_conv1_2'])
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
  tower_out, args['tower_out'] = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False, namepre=namepre+'_tower_out', args=args['tower_out'])
  net = net + scale * tower_out
  act = net
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}, namepre='', args=None):
  if args is None:
    args = {'tower_conv':None, 'tower_conv1_0':None, 'tower_conv1_1':None, 'tower_conv1_2':None, 'tower_out':None}
  tower_conv, args['tower_conv'] = ConvFactory(net, 192, (1, 1), namepre=namepre+'_tower_conv', args=args['tower_conv'])
  tower_conv1_0, args['tower_conv1_0'] = ConvFactory(net, 192, (1, 1), namepre=namepre+'_tower_conv1_0', args=args['tower_conv1_0'])
  tower_conv1_1, args['tower_conv1_1'] = ConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1), namepre=namepre+'_tower_conv1_1', args=args['tower_conv1_1'])
  tower_conv1_2, args['tower_conv1_2'] = ConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0), namepre=namepre+'_tower_conv1_2', args=args['tower_conv1_2'])
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
  tower_out, args['tower_out'] = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False, namepre=namepre+'_tower_out', args=args['tower_out'])
  net = net + scale * tower_out
  act = net
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr, name=namepre+'_act')
  return act, args


def repeat(inputs, repetitions, layer, *ltargs, **kwargs):
  outputs = inputs
  namepre = kwargs['namepre']
  args = kwargs['args']
  if args is None:
    args = {}
    for i in xrange(repetitions):
      argname='repeat_'+str(i)
      args[argname] = None
  for i in range(repetitions):
    kwargs['namepre'] = namepre+'_'+str(i)
    argname='repeat_'+str(i)
    kwargs['args'] = args[argname]
#    print ltargs
#    print kwargs
    outputs, args[argname] = layer(outputs, *ltargs, **kwargs)

  return outputs, args


def create_inception_resnet_v2(data, namepre='', args=None):
  if args is None:
    args = {'stem':None, 'reductionA':None, 'repeat_block35':None, 'reductionB':None, 
            'repeat_block17':None, 'reductionC':None, 'repeat_block8':None, 
            'final_block8':None, 'final_conv':None, 'finalfc':None}

  stem_net, args['stem']= stem(data, namepre=namepre+'_stem', args=args['stem'])

  reduceA, args['reductionA'] = reductionA(stem_net, namepre=namepre+'_reductionA', args=args['reductionA'])

  repeat_block35, args['repeat_block35'] = repeat(reduceA, 2, block35, scale=0.17, input_num_channels=320, namepre=namepre+'_repeat_block35', args=args['repeat_block35'])


  reduceB, args['reductionB'] = reductionB(repeat_block35, namepre=namepre+'_reductionB', args=args['reductionB'])

  repeat_block17, args['repeat_block17'] = repeat(reduceB, 4, block17, scale=0.1, input_num_channels=1088, namepre=namepre+'_repeat_block17', args=args['repeat_block17'])

  reduceC, args['reductionC'] = reductionC(repeat_block17, namepre=namepre+'_reductionC', args=args['reductionC'])

  repeat_block8, args['repeat_block8'] = repeat(reduceC, 2, block8, scale=0.2, input_num_channels=2080, namepre=namepre+'_repeat_block8', args=args['repeat_block8'])
  final_block8, args['final_block8'] = block8(repeat_block8, with_act=False, input_num_channels=2080, namepre=namepre+'_final_block8', args=args['final_block8'])

  final_conv, args['final_conv'] = ConvFactory(final_block8, 1536, (1, 1), namepre=namepre+'_final_conv', args=args['final_conv'])
  final_pool = mx.symbol.Pooling(final_conv, kernel=(8, 8), global_pool=True, pool_type='avg', name=namepre+'_final_pool')
#   final_pool = mx.symbol.Pooling(final_conv, kernel=(5, 5), stride=(1, 1), pool_type='avg', name=namepre+'_final_pool')
  final_flatten = mx.symbol.Flatten(final_pool, name=namepre+'_final_flatten')

  drop1 = mx.sym.Dropout(data=final_flatten, p=0.5, name=namepre+'_dropout1')

  if args['finalfc'] is None:
    args['finalfc'] = {}
    args['finalfc']['weight'] = mx.sym.Variable(namepre+'_fc1_weight')
    args['finalfc']['bias'] = mx.sym.Variable(namepre+'_fc1_bias')
    
  reid_fc1 = mx.sym.FullyConnected(data=drop1, num_hidden=FEAT_DIM, name=namepre+"_fc1", 
                                   weight=args['finalfc']['weight'], bias=args['finalfc']['bias']) 
#  reid_act = mx.sym.Activation(data=reid_fc1, act_type='tanh', name=namepre+'_fc1_relu')

  net = reid_fc1
#  net = final_flatten

  return net, args

def create_net(data, radius):
#     data = mx.sym.Variable('data')
    args_all = None
    feat_final, args_all = create_inception_resnet_v2(data, namepre='part1', args=args_all)
    feat_final = mx.sym.BatchNorm(data=feat_final, fix_gamma=False, name='feat_bn1')


    min_value =10**-36
    norm_value = radius
    #  norm_value = 24
#     logging.info('norm_value:%f, min_value:%e, hardratio:%f', norm_value, min_value, hardratio)
    logging.info('norm_value:%f, min_value:%e', norm_value, min_value)
    #norm
    znorm_loss = None
    if norm_value>0:
    #    proxy_Z = mx.sym.L2Normalization(proxy_Z) * norm_value
    #    feat_final = mx.sym.L2Normalization(feat_final) * norm_value
#         proxy_Znorm = mx.sym.sum_axis(proxy_Z**2, axis=1)
#         proxy_Znorm = mx.sym.sqrt(proxy_Znorm) + min_value

#     #    znorm_loss = mx.sym.abs(proxy_Znorm - 1.0)
#     #    znorm_loss = mx.sym.sum(znorm_loss)
#     #    znorm_loss = mx.sym.MakeLoss(znorm_loss)

#         proxy_Znorm = mx.sym.Reshape(proxy_Znorm, shape=(-2, 1))
#         proxy_Z = mx.sym.broadcast_div(proxy_Z, proxy_Znorm)# * norm_value

        feat_finalnorm = mx.sym.sum_axis(feat_final**2, axis=1)
        feat_finalnorm = mx.sym.sqrt(feat_finalnorm) + min_value
        feat_finalnorm = mx.sym.Reshape(feat_finalnorm, shape=(-2, 1))
        feat_final = mx.sym.broadcast_div(feat_final, feat_finalnorm) * norm_value
#     X = mx.nd.empty(shape=(64, 10000000), ctx=mx.cpu(0))
# mx.sym.Variable(name='proxy_Z_weight', shape=(proxy_num, featdim), dtype=np.float32)
    return feat_final


SNAP_SHOT_PATH = '/train/execute/'
EXECUTE_PATH = '/train/execute/'
def load_checkpoint(model, prefix, epoch):
    param_name = SNAP_SHOT_PATH + '%s-%04d.params' % (prefix, epoch)
    save_dict = mx.nd.load(param_name)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if name=='proxy_Z_weight':
#             sp = pZshape
#             rndv = np.random.rand(*sp)-0.5
#             arg_params[name] = mx.nd.array(rndv)
            print 'skipped %s...'%name
            continue
        if arg_type == 'arg':
            arg_params[name] = value
        elif arg_type == 'aux':
            aux_params[name] = value
        else:
            raise ValueError("Invalid param file " + fname)
    model.set_params(arg_params, aux_params, allow_missing=True)
    arg_params, aux_params = model.get_params()
    print 'Load checkpoint from \"%s\"'%(param_name) 
    return model, arg_params, aux_params


import ctypes
from mxnet.base import check_call, _LIB
def get_pointer_value(v):
#     print 'run.......'
    cp = ctypes.c_void_p() 
    _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    return cp.value
# def cublas_sgemm(X, W, handle, cpu=True):
#     '''
#         Input:
#             X     : mxnet ndarray, shape=(N,C), dtype=float32
#             W     : mxnet ndarray, shape=(C,M), dtype=float32
#             handle: cublasXt handle
#         Output:
#             score : mxnet ndarray, shape=(N,M), dtype=float32
#     '''
#     N, C = X.shape
#     M = W.shape[1]
# #     if cpu:
#     score = mx.nd.empty(shape=(M,N),ctx=mx.cpu(), dtype='float32')
# #     else:
# #         score = mx.nd.empty(shape=(M,N),ctx=mx.gpu(0), dtype='float32')
# #     print type(get_pointer_value(score))
#     def stream_cal_in_GPU():        
#         cublasxt.cublasXtSgemm(handle,
#                                'C', 'C', 
#                                N, M, C, np.float32(1.0),
#                                get_pointer_value(X), C, get_pointer_value(W), M, np.float32(0.0), get_pointer_value(score), N)
#     cublasxt.cublasXtSetCpuRoutine(handle, 0, 0, stream_cal_in_GPU())
#     return score.T

class mixModule(object):
    def __init__(self, symbol, context, handle, hardratio, data_shape, proxy_Z, K, param_prefix, load_paramidx = None):

        self.mod = mx.mod.Module(symbol=symbol, 
                                 data_names=("data",), 
                                 label_names=None, 
                                 context=context)
        self.mod.bind(data_shapes=[("data", data_shape)])
#         self.mod.set_params(arg_params, aux_params, allow_missing=True)
        if load_paramidx is not None:
#             symbol, arg_params, aux_params = mx.model.load_checkpoint(param_prefix, load_paramidx)
            self.mod, arg_params, aux_params = load_checkpoint(self.mod, param_prefix, load_paramidx)
            print 'fine-tuning...'
        
        self.context = context if isinstance(context, list) else [context]
        self.handle = handle
        self.hardratio = hardratio
        self.W = proxy_Z.T
        self.N = data_shape[0]
        self.C, self.M = self.W.shape
        self.score = mx.nd.empty(shape=(self.M, self.N), dtype='float32', ctx=mx.cpu())
        self.X = mx.nd.empty(shape=(self.N, self.C), dtype='float32', ctx=mx.gpu(0))
        self.K = K
#         self.history = mx.nd.zeros(shape=(self.C, self.M), ctx=mx.cpu(), dtype='float32')
        self.ang = None
#         self.history = np.zeros((self.C, self.M), np.float32)
        self.loss = None
    def init_params(self, *args, **kwargs):
        self.mod.init_params(*args, **kwargs)

    def init_optimizer(self, *args, **kwargs):
        self.mod.init_optimizer(*args, **kwargs)
        
    def save_checkpoint(self, prefix, epoch):
#         self.mod.symbol.save(SNAP_SHOT_PATH + '%s-symbol.json' % prefix)
#         param_name = SNAP_SHOT_PATH + '%s-%04d.params' % (prefix, epoch)
        self.mod.symbol.save('%s-symbol.json' % prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)        
        self.mod.save_params(param_name)
        print 'Saved checkpoint to \"%s\"'%(param_name)
    
    def update(self, data_batch):
        self.mod.forward(data_batch)
        self.X = self.mod.get_outputs()[0]
        self.X.wait_to_read()
#         print 'self.X : ', self.X[0,:10]
#         mx.nd.save('feat_final.params', self.X)
#         self.X = mx.nd.array(np.load('../datatemp/feat_final.npy'), dtype='float32', ctx=mx.gpu(0))
#         print 'feat_final : ', self.X[0,:10]
        y_true = data_batch.label[0].reshape(-1,1)
        y_true = y_true.astype('int32')
#         print 'label : ', y_true[:10].reshape(-1)
        self.W = self.W / (mx.nd.sqrt(mx.nd.sum(self.W**2, axis=0, keepdims=True)) + 10**-36) # M * C
#         print 'self.W shape: ', self.W.shape
#         print 'self.W : ', self.W[0, :10]
        self.W.wait_to_read()
        def stream_cal_in_GPU():        
            cublasxt.cublasXtSgemm(self.handle,
                                   'C', 'C', 
                                   self.N, self.M, self.C, np.float32(1.0),
                                   get_pointer_value(self.X), self.C, get_pointer_value(self.W), self.M, np.float32(0.0),
                                   get_pointer_value(self.score), self.N)
        cublasxt.cublasXtSetCpuRoutine(self.handle, 0, 0, stream_cal_in_GPU())
#         print 
        self.score = self.score.T
        self.score.wait_to_read()
#         print 'self.score : ', self.score[0,:10]
#         mx.nd.save('score.params', self.score)
        self.ang = self.score[range(self.N), y_true.reshape(-1)]
        self.score[range(self.N), y_true.reshape(-1)] += np.log(self.hardratio)
        
#         print 'self.score : ', self.score[0,:10]
        self.score = self.score - mx.nd.max(self.score, axis=1, keepdims=True)
        self.score = mx.nd.exp(self.score)
        self.score = self.score / mx.nd.sum(self.score, axis=1, keepdims=True)
#         print self.score.shape
        y_true_score = self.score[range(self.N), y_true.reshape(-1)]
        self.score[range(self.N), y_true.reshape(-1)] = self.score.min(axis = 1)
        
        TopK_idx = mx.nd.topk(data=self.score, axis=1, k=self.K).astype('int32')
        self.score[range(self.N), y_true.reshape(-1)] = y_true_score
        TopK_idx = mx.nd.concat(TopK_idx, y_true, dim=1)
        
        TopK_score = self.score[np.array(range(self.N)).reshape(-1,1), TopK_idx]
        y_true_reform = mx.nd.array([self.K]*self.N).astype('int32').reshape(-1,1)
        
        def softmax_loss(sparse_probs, y_true):
            """
            Computes the loss and gradient for softmax classification.
            Inputs:
            - sparse_probs: Input data, of shape (N, K) where sparse_probs[i, j] is the probability for the jth
              class for the ith input.
            - y_true: Vector of labels, of shape (N,1) where y_true[i] is the label for probs[i] and
              0 <= y_true[i] < K
            Returns a tuple of:
            - loss: Scalar giving the loss
            - d_sparse_score: shape: (N, K), Gradient of the loss with respect to sparse_score (not sparse_probs)
            """
            N = sparse_probs.shape[0]
        #     # Numerical stability
        #     shifted_sparse_score = sparse_score - np.max(sparse_score, axis=1, keepdims=True)

        #     Z = np.sum(np.exp(shifted_sparse_score), axis=1, keepdims=True)
        #     log_probs = shifted_sparse_score - np.log(Z)
        #     probs = np.exp(log_probs)
            log_sparse_probs = mx.nd.log(sparse_probs)
            loss = -mx.nd.sum(log_sparse_probs[np.arange(N), y_true.reshape(-1)]) / N
            d_sparse_score = sparse_probs.copy()
            d_sparse_score[np.arange(N), y_true.reshape(-1)] -= 1
            # rescale gradient
            d_sparse_score /= N
            return loss, d_sparse_score
        
        self.loss, d_TopK_score = softmax_loss(TopK_score, y_true_reform)
#         print 'loss: ', self.loss
        self.score[:] = 0
        self.score[np.array(range(self.N)).reshape(-1,1), TopK_idx] = d_TopK_score
        
#         d_score = mx.nd.array(self.score, dtype='float32')
        csr_d_score = self.score.tostype('csr')
        self.score = self.score.T
        
        X_cpu = self.X.T.copyto(mx.cpu())
        d_proxy_Z = mx.ndarray.sparse.dot(X_cpu, csr_d_score)
#         print csr_d_score.shape, self.X.shape
#         print d_proxy_Z.shape
        d_feat_final = mx.nd.dot(csr_d_score, self.W.T)

        d_feat_final = mx.nd.array(d_feat_final, dtype='float32', ctx=mx.gpu(0))
        # backprop
        self.mod.backward([d_feat_final])

        # update W
        num_iter = self.mod._optimizer.num_update
        lr = self.mod._optimizer._get_lr(num_iter)
#         wd = self.mod._optimizer._get_wd(num_iter)
        self.W += -lr * d_proxy_Z
        
#         wd = self.mod._optimizer._get_wd(num_iter)
#         eps = self.mod._optimizer.epsilon
# #         print 'iter: %d, lr: %e' % (num_iter, lr)
#         self.history[:] += (d_proxy_Z**2)
# #         proxy_Z[:] += -lr * (d_proxy_Z / (self.history + eps).sqrt() + wd * proxy_Z)
#         self.W[:] += -lr * (d_proxy_Z / np.sqrt(self.history + eps) + wd * self.W)
    
    
#         self.W = proxy_Z.asnumpy()
#         self.W = proxy_Z
        # update module
        self.mod.update()
    def save_proxy_Z(self, proxy_Z_fn):   
        # save proxy_Z(W) after a certain number of self.mod._optimizer.num_update
#         if num_iter%1000 == 0:
        mx.nd.save(proxy_Z_fn, self.W)
        print 'Save proxy_Z in "%s"' % (proxy_Z_fn)
    def get_loss(self):
        return self.loss
    
    def get_ang(self, radius):
        self.ang /= radius
        return mx.nd.mean(mx.nd.arccos(self.ang)/np.pi*180)