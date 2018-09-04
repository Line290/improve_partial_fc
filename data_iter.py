import mxnet as mx
import time

def DataIter(prefix = '/train/trainset/list_clean_28w_20180803', image_shapes=(64, 3, 240, 120), data_nthreads = 8):
#     prefix = '/train/trainset/list_clean_28w_20180803'
    image_shape = (image_shapes[1], image_shapes[2], image_shapes[3])
    batch_size = image_shapes[0]
    # data_nthreads = 8
    train = mx.io.ImageRecordIter(
            path_imgrec         = prefix + '.rec',
            path_imgidx         = prefix + '.idx',
            label_width         = 1,
#             mean_r              = 127.5,
#             mean_g              = 127.5,
#             mean_b              = 127.5,
    #         std_r               = rgb_std[0],
    #         std_g               = rgb_std[1],
    #         std_b               = rgb_std[2],
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = image_shape,
            batch_size          = batch_size,
            resize              = max(image_shapes[2], image_shapes[3]),
    #         rand_crop           = args.random_crop,
    #         max_random_scale    = args.max_random_scale,
    #         pad                 = args.pad_size,
    #         fill_value          = args.fill_value,
    #         random_resized_crop = args.random_resized_crop,
    #         min_random_scale    = args.min_random_scale,
    #         max_aspect_ratio    = args.max_random_aspect_ratio,
    #         min_aspect_ratio    = args.min_random_aspect_ratio,
    #         max_random_area     = args.max_random_area,
    #         min_random_area     = args.min_random_area,
    #         min_crop_size       = args.min_crop_size,
    #         max_crop_size       = args.max_crop_size,
    #         brightness          = args.brightness,
    #         contrast            = args.contrast,
    #         saturation          = args.saturation,
    #         pca_noise           = args.pca_noise,
    #         random_h            = args.max_random_h,
    #         random_s            = args.max_random_s,
    #         random_l            = args.max_random_l,
    #         max_rotate_angle    = args.max_random_rotate_angle,
    #         max_shear_ratio     = args.max_random_shear_ratio,
    #         rand_mirror         = args.random_mirror,
            preprocess_threads  = data_nthreads,
            shuffle             = True,
            scale               = 1.0/255
    #         num_parts           = nworker,
    #         part_index          = rank,
    )
    return train
# print dir(train)
if __name__ == '__main__':
    start = time.time()
    data_iter = DataIter()
    for i in range(1000):
        data_batch = data_iter.next()
    print 'cost time: %e' % (time.time()-start)
    print data_batch.data[0].shape
    print data_batch.label[0].shape
    data_iter.reset()