import os
import time
import argparse

import tensorflow as tf
import glob
import models as net
from config_utils import load_configfile, check_trainconfiguration

# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save_inference_model(config_file, input_model, output_model, model='blstm'):
    config = load_configfile(config_file)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            if model == 'blstm':
                config = check_trainconfiguration(config)
    
                with tf.name_scope('placeholder'):
                    sequence_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
                    target_sources_ph = tf.placeholder(tf.float32, shape=[None, config['audio_len']], name='target_sources')
                    video_features_ph = tf.placeholder(tf.float32, shape=[None, None, config['video_feat_dim']], name='video_features')
                    masks_ph = tf.placeholder(tf.float32, shape=[None, None, config['audio_feat_dim']], name='masks')
                    audio_feat_mean_ph = tf.placeholder(tf.float32, shape=[config['audio_feat_dim']], name='audio_features_mean')
                    audio_feat_std_ph = tf.placeholder(tf.float32, shape=[config['audio_feat_dim']], name='audio_features_std')
                    dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate')
    
                print('Building speech inpainting inference model:')
                netname='SINet'
                with tf.variable_scope(netname):
                    sinet = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
                    sinet.build_graph(var_scope=netname)
                print('Model building done.')
                
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            saverInputNet = tf.train.Saver(var_list=train_vars)
            saverOutRecNet = tf.train.Saver()
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                sess.run(init_op)
                saverInputNet.restore(sess, input_model)
                spath = saverOutRecNet.save(sess, output_model)
                print("Model saved in", spath)
