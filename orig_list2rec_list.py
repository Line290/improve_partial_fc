# from __future__ import print_function
import os
import sys

# curr_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
sys.path.insert(0, '/train/execute/distribution')
import random
import argparse
import cv2
import time
import traceback

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

import DataGenerator as dg

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '%f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)
            
def make_list_new(image_list, prefix):
#     image_list = list_image(args.root, args.recursive, args.exts)
    # get image list
    # (No, path, ID or label)
    # e.g. (0, '201709011900/10.209.2.85-0-201709011900-201709012200/1/1_1_1.jpg', 0)
#     image_list = list(image_list)
    shuffle = True
    if shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    
    # split several blocks, the number is chunks
#     chunk_size = (N + args.chunks - 1) // args.chunks
    chunk_size = N
#     for i in range(args.chunks):
    for i in range(1):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
#         if 1 > 1:
#             str_chunk = '_%d' % i
#         else:
        str_chunk = ''
        sep = int(chunk_size * 1)
#         sep = int(chunk_size * args.train_ratio)
#         sep_test = int(chunk_size * args.test_ratio)
#         if args.train_ratio == 1.0:
#         prefix = '/train/trainset/list_28w_20180731'
        write_list(prefix + str_chunk + '.lst', chunk)
#         else:
#             if args.test_ratio:
#                 write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
#             if args.train_ratio + args.test_ratio < 1.0:
#                 write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
#             write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])



# list_path = './listFolder/trainlist_reid/list_all_20180723.list'
# all_lists = dg.get_datalist2([list_path])
# print all_lists[0]

# lists = []
# for i, onelist in enumerate(all_lists):
#     path, name, img_id = onelist.split('*')
# #     print path, name, img_id
# #     break
#     path = path + '/' + name
#     newline = (i, path, int(img_id))
#     lists.append(newline)
if __name__ == '__main__':
    
    list_path = '/train/execute/listFolder/trainlist_reid/list_clean_28w_20180803.list'
    all_lists = dg.get_datalist2([list_path])
    
    lists = []
    for i, onelist in enumerate(all_lists):
        path, name, img_id = onelist.split('*')
    #     print path, name, img_id
    #     break
        path = path + '/' + name
        newline = (i, path, int(img_id))
        lists.append(newline)
    
    prefix = '/train/trainset/list_clean_28w_20180803'    
    make_list_new(lists, prefix)
