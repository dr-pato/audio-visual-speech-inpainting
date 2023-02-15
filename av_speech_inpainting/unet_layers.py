from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf


def encoder_layer_fconv(x, filter_size, num_features, num_filters, stride=1, max_pool=None, batch_norm=True, activation=True, is_training=False):
        stddev = tf.math.sqrt(2 / (filter_size ** 2 * num_filters))
        w = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_features, num_filters], stddev=stddev), name='w')
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

        conv_2d = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        conv_2d = tf.nn.bias_add(conv_2d, b)
        if batch_norm:
            conv_2d = tf.layers.batch_normalization(conv_2d, training=is_training)
        if activation:
            conv_2d = tf.nn.relu(conv_2d)
        if max_pool is not None:
            conv_2d = tf.nn.max_pool(conv_2d, ksize=[1, max_pool, max_pool, 1], strides=[1, max_pool, max_pool, 1], padding='SAME')

        return conv_2d


def decoder_layer_fconv(x, e_conv, filter_size, num_features, num_filters, stride=1, batch_norm=True, activation=True, is_training=False):
    stddev = tf.math.sqrt(2 / (filter_size ** 2 * num_filters))
    w = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_features, num_filters], stddev=stddev), name='w')
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

    up_x = tf.keras.layers.UpSampling2D((2,2))(x)
    concat_x = tf.concat([e_conv, up_x], axis=3)
    conv_2d = tf.nn.conv2d(concat_x, w, strides=[1, stride, stride, 1], padding='SAME')
    conv_2d = tf.nn.bias_add(conv_2d, b)
    if batch_norm:
        conv_2d = tf.layers.batch_normalization(conv_2d, training=is_training)
    if activation:
        conv_2d = tf.nn.leaky_relu(conv_2d, alpha=0.2)

    return conv_2d


def encoder_layer_pconv(x, mask, filter_size, num_features, num_filters, stride=2, batch_norm=True, activation=True, is_training=False):
        # input convolution
        stddev = tf.math.sqrt(2 / (filter_size ** 2 * num_filters))
        w = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_features, num_filters], stddev=stddev), name='w')
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
        x_out = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

        # mask convolution
        kernel_mask = tf.ones([filter_size, filter_size, num_features, num_filters])
        mask_out = tf.nn.conv2d(mask, kernel_mask, strides=[1, stride, stride, 1], padding='SAME')

        # mask ratio calculatior for out normalization
        mask_ratio = (filter_size ** 2) / (mask_out + 1e-8)

        # apply bias
        x_out = tf.nn.bias_add(conv_2d, b)
        if batch_norm:
            conv_2d = tf.layers.batch_normalization(conv_2d, training=is_training)
        if activation:
            conv_2d = tf.nn.relu(conv_2d)
        
        return conv_2d

 
def decoder_layer_pconv(x, e_conv, filter_size, num_features, num_filters, stride=1, batch_norm=True, activation=True, is_training=False):
    stddev = tf.math.sqrt(2 / (filter_size ** 2 * num_filters))
    w = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_features, num_filters], stddev=stddev), name='w')
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

    up_x = tf.keras.layers.UpSampling2D((2,2))(x)
    concat_x = tf.concat([e_conv, up_x], axis=3)
    conv_2d = tf.nn.conv2d(concat_x, w, strides=[1, stride, stride, 1], padding='SAME')
    conv_2d = tf.nn.bias_add(conv_2d, b)
    if batch_norm:
        conv_2d = tf.layers.batch_normalization(conv_2d, training=is_training)
    if activation:
        conv_2d = tf.nn.leaky_relu(conv_2d, alpha=0.2)

    return conv_2d


def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)

def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME', name="conv2d_transpose")

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
