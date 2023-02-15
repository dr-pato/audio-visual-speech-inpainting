from __future__ import division

import os
import sys
import shutil
import numpy as np
from glob import glob
from time import time
import random
from dataset_reader_emb import DataManager
import models as net
from config_utils import load_configfile, check_trainconfiguration

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')

# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train(config_file):
    """
    Train the speech inpainting model.
    """
    # Load and check configuration file
    config = load_configfile(config_file)
    config = check_trainconfiguration(config)

    data_path = config['root_folder']
    data_path_train = os.path.join(data_path, 'training-set')
    data_path_val = os.path.join(data_path, 'validation-set')

    exp_path = config['exp_folder']
    exp_name = os.path.basename(exp_path)
    checkpoints_dir = os.path.join(exp_path, 'netmodel')
    tensorboard_dir = os.path.join(exp_path, 'tfboard')
    training_log_file = os.path.join(exp_path, 'training_log.txt')

    # Training Graph
    with tf.Graph().as_default():
        # Create the DataManager that reads training and validation TFRecords.
        with tf.name_scope('train_batch'):
            train_data_manager = DataManager(num_audio_samples=config['audio_len'], audio_feat_size=config['audio_feat_dim'],
                                             video_feat_size=config['video_feat_dim'], embedding_size=512, buffer_size=4000, mode='fixed')
            train_files_list = glob(os.path.join(data_path_train, '*.tfrecord'))
            random.shuffle(train_files_list)
            train_dataset = train_data_manager.get_dataset(train_files_list, shuffle=True)
            train_batch_dataset, train_it = train_data_manager.get_iterator(train_dataset, batch_size=config['batch_size'],
                                                                            n_epochs=1, drop_remainder=False)
            next_train_batch = train_it.get_next()
        with tf.name_scope('validation_batch'):
            val_data_manager = DataManager(num_audio_samples=config['audio_len'], audio_feat_size=config['audio_feat_dim'],
                                           video_feat_size=config['video_feat_dim'], embedding_size=512, buffer_size=4000, mode='fixed')
            val_files_list = glob(os.path.join(data_path_val, '*.tfrecord'))
            val_dataset = val_data_manager.get_dataset(val_files_list, shuffle=False)
            val_batch_dataset, val_it = val_data_manager.get_iterator(val_dataset, batch_size=config['batch_size'],
                                                                      n_epochs=1, drop_remainder=False)
            next_val_batch = val_it.get_next()

        # Load normalization data
        audio_feat_mean = np.load(config['audio_feat_mean'])
        audio_feat_std = np.load(config['audio_feat_std'])
        
        # Placeholders
        with tf.name_scope('placeholder'):
            sequence_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
            target_sources_ph = tf.placeholder(tf.float32, shape=[None, config['audio_len']], name='target_sources')
            video_features_ph = tf.placeholder(tf.float32, shape=[None, None, config['video_feat_dim']], name='video_features')
            embeddings_ph = tf.placeholder(tf.float32, shape=[None, 512], name='embeddings')
            masks_ph = tf.placeholder(tf.float32, shape=[None, None, config['audio_feat_dim']], name='masks')
            audio_feat_mean_ph = tf.placeholder(tf.float32, shape=[config['audio_feat_dim']], name='audio_features_mean')
            audio_feat_std_ph = tf.placeholder(tf.float32, shape=[config['audio_feat_dim']], name='audio_features_std')
            dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate')
            
        # Graph building and definition.
        if config['model'] == 'av-blstm-twosteps':
            model = net.StackedBLSTM2StepsModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                audio_feat_std_ph, dropout_rate_ph, config, video_features=video_features_ph)
            model.build_graph(var_scope=config['model'])
        else:
            with tf.variable_scope(config['model']):
                if config['model'] == 'a-blstm':
                    model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, input='a', is_training=False)
                elif config['model'] == 'v-blstm':
                    model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
                elif config['model'] == 'av-blstm':
                    model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
                elif config['model'] == 'a-blstm-ssnn':
                    model = net.StackedBLSTMSSNNModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                      audio_feat_std_ph, dropout_rate_ph, config, input='a')
                elif config['model'] == 'v-blstm-ssnn':
                    model = net.StackedBLSTMSSNNModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                      audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
                elif config['model'] == 'av-blstm-ssnn':
                    model = net.StackedBLSTMSSNNModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                      audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
                elif config['model'] == 'a-blstm-emb':
                    model = net.StackedBLSTMEmbeddingModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, embeddings=embeddings_ph, input='a', is_training=False)
                elif config['model'] == 'v-blstm-emb':
                    model = net.StackedBLSTMEmbeddingModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, embeddings=embeddings_ph, input='v', video_features=video_features_ph)
                elif config['model'] == 'av-blstm-emb':
                    model = net.StackedBLSTMEmbeddingModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, embeddings=embeddings_ph, input='av', video_features=video_features_ph)
                elif config['model'] == 'unet':
                    model = net.UNetFConvModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                               audio_feat_std_ph, dropout_rate_ph, config)
                else:
                    print('Model selection must be "a-blstm", "v-blstm", "av-blstm", "av-blstm-twosteps" or "unet". Closing...')
                    sys.exit(1)
                model.build_graph(var_scope=config['model'])
        print('Model building done.')

        # The inizializer operation.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Save and restore all the variables.
        saver = tf.train.Saver()
        if config['model'] == 'av-blstm-twosteps':
            saver_vnet = tf.train.Saver(var_list=model.video_model.train_vars)

        # Create log directories if not exist.
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        # Copy configuration file in experiment folder
        dest_config_file = os.path.join(checkpoints_dir, 'config.txt')
        if dest_config_file != config_file:
            shutil.copy(config_file, dest_config_file)
        # Copy normalization data in experiment folder
        shutil.copy(config['audio_feat_mean'], os.path.join(checkpoints_dir, 'audio_features_mean.npy'))
        shutil.copy(config['audio_feat_std'], os.path.join(checkpoints_dir, 'audio_features_std.npy'))
        
        # Open training log file
        training_log_file = open(training_log_file, 'a')

        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            # FileWriter to save TensorBoard summary
            tb_writer = tf.summary.FileWriter(tensorboard_dir, graph=sess.graph)
            
            # Initialize the variables and the batch iterator.
            sess.run(init_op)
            
            # Estimate number of steps per epoch
            # Load training sequence lengths from NPY file
            seq_lengths_train = np.load(os.path.join(data_path_train, 'seq_lengths.npy'))
            seq_lengths_val = np.load(os.path.join(data_path_val, 'seq_lengths.npy'))
            train_size = len(seq_lengths_train)
            val_size = len(seq_lengths_val)
            n_steps_epoch = int(train_size / config['batch_size'])
            n_steps = n_steps_epoch * config['max_n_epochs']

            # Restore variables
            if config['model_ckp_vnet']:
                try:
                    saver_vnet.restore(sess, config['model_ckp_vnet'])
                    print('Visual model variables restored.')
                except ValueError:
                    print('{:s} is not a valid checkpoint. Closing...'.format(config['model_ckp_vnet']))
                    exit(2)
            if config['model_ckp']:
                try:
                    saver.restore(sess, config['model_ckp'])
                    print('Model variables restored.')
                except ValueError:
                    print('{:s} is not a valid checkpoint. Closing...'.format(config['model_ckp']))
                    exit(2)
            else:
                training_log_file.write('+-- EXPERIMENT NAME - {:s} --+\n'.format(exp_name))
                training_log_file.write('## Model type: {:s}\n'.format(config['model']))
                training_log_file.write('## Network dimensions: {:s}\n'.format(str(config['net_dim'])))
                training_log_file.write('## Optimizer: {:s}\n'.format(config['optimizer_type']))
                training_log_file.write('## Starter learning rate: {:.6f}\n'.format(config['starter_learning_rate']))
                training_log_file.write('## Learning rate update steps: {:d}\n'.format(config['lr_updating_steps']))
                training_log_file.write('## Learning rate decay: {:.6f}\n'.format(config['lr_decay']))
                training_log_file.write('## L2 regularization coefficient: {:.6f}\n'.format(config['l2']))
                training_log_file.write('## Dropout rate (no dropout if 0): {:.6f}\n'.format(config['dropout_rate']))
                training_log_file.write('## Training dataset: {:s}\n'.format(data_path_train))
                training_log_file.write('## Training size: {:d}\n'.format(train_size))
                training_log_file.write('## Validation dataset: {:s}\n'.format(data_path_val))
                training_log_file.write('## Validation size: {:d}\n'.format(val_size))
                training_log_file.write('## Batch size: {:d}\n'.format(config['batch_size']))
                training_log_file.write('## Approximated number of steps per epoch: {:d}\n'.format(n_steps_epoch))
                training_log_file.write('## Number of training epochs: {:d}\n'.format(config['max_n_epochs']))
                training_log_file.write('## Approximated total number of steps: {:d}\n'.format(n_steps))
                training_log_file.write('\nEpoch\tLR\tTraining loss\tValidation loss\t[TIME]\n')
            
            # Print training details.
            print('')
            print('+-- EXPERIMENT NAME - {:s} --+'.format(exp_name))
            print('## Model type: {:s}'.format(config['model']))
            print('## Network dimensions: {:s}'.format(str(config['net_dim'])))
            print('## Optimizer: {:s}'.format(config['optimizer_type']))
            print('## Starter learning rate: {:.6f}'.format(config['starter_learning_rate']))
            print('## Learning rate update steps: {:d}'.format(config['lr_updating_steps']))
            print('## Learning rate decay: {:.6f}'.format(config['lr_decay']))
            print('## L2 regularization coefficient: {:.6f}'.format(config['l2']))
            print('## Dropout rate (no dropout if 0): {:.6f}'.format(config['dropout_rate']))
            print('## Training dataset: {:s}'.format(data_path_train))
            print('## Training size: {:d}'.format(train_size))
            print('## Validation dataset: {:s}'.format(data_path_val))
            print('## Validation size: {:d}'.format(val_size))
            print('## Batch size: {:d}'.format(config['batch_size']))
            print('## Approximated number of steps per epoch: {:d}'.format(n_steps_epoch))
            print('## Number of training epochs: {:d}'.format(config['max_n_epochs']))
            print('')

            tot_step = sess.run(model.global_step)
            epoch_counter = int(tot_step / n_steps_epoch)
            best_val_checkpoint = (0, 0) # Save best model on validation set
            best_val_loss = -1.0 # Save best validation Frame Error Rate
            cneg_epochs = 0 # Variable used for early stopping
            train_start_time = time()
            
            for n_epoch in range(config['max_n_epochs']):
                epoch_counter += 1
                n_step = 0
                epoch_start_time = time()
                
                # Training set iterator initialization
                sess.run(train_it.initializer)
                
                print('-> Epoch {:d}'.format(epoch_counter))
                while True:
                    try:
                        n_step += 1
                        tot_step += 1

                        # Fetch training batch.
                        length_batch, length_lab_batch, target_audio_batch, embeddings_batch, \
                            sample_path_batch, labels_batch, video_batch, mask_batch = sess.run(next_train_batch)
                        max_seq_len = max(length_batch)
                        
                        loss, loss_fn, lr, _ = sess.run(fetches=[model.loss, model.loss_func, model.learning_rate, model.train_op],
                                            feed_dict={
                                              sequence_lengths_ph: length_batch,
                                              target_sources_ph: target_audio_batch,
                                              video_features_ph: video_batch,
                                              embeddings_ph: embeddings_batch,
                                              masks_ph: mask_batch,
                                              audio_feat_mean_ph: audio_feat_mean,
                                              audio_feat_std_ph: audio_feat_std,
                                              dropout_rate_ph: config['dropout_rate']
                                              })
                        
                        if np.isnan(loss):
                            print('GOT INSTABILITY: loss is NaN. Leaving...')
                            sys.exit(1)
                        if np.isinf(loss):
                            print('GOT INSTABILITY: loss is inf. Leaving...')
                            sys.exit(1)
                        
                            
                        if n_step == 1:
                            nframe_sum = np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']
                            train_avg_loss = loss
                            train_avg_loss_fn = loss_fn
                        else:
                            prev_nframe_sum = nframe_sum
                            nframe_sum += np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']
                            train_avg_loss = (train_avg_loss * prev_nframe_sum + loss * np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']) / nframe_sum
                            train_avg_loss_fn = (train_avg_loss_fn * prev_nframe_sum + loss_fn * np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']) / nframe_sum
                            
                        # Print loss value fairly often.
                        if n_step % 200 == 0 or n_step == 1:
                            print('Step[{:7d}] Loss[{:3.5f}|{:3.5f}] LR[{:.6f}] Epoch training time[{:.2f}]'. \
                                format(tot_step, train_avg_loss, train_avg_loss_fn, lr, time() - epoch_start_time))
                        if n_step % 1000 == 0:
                            save_path = saver.save(sess, os.path.join(checkpoints_dir, 'ckpt'))
                            print('Model checkpoint saved in file %s' % save_path)

                    except tf.errors.OutOfRangeError:
                        print('Completed epoch {:d} at step {:d} --> Training loss: {:3.5f} - {:3.5f}'. \
                            format(epoch_counter, tot_step, train_avg_loss, train_avg_loss_fn))
                        epoch_duration = time() - epoch_start_time
                        print('Epoch training time (seconds) = {:.6f}'.format(epoch_duration))
                        break


                print('Start validation set evaluation...')
                # Validation set iterator initialization
                sess.run(val_it.initializer)
                
                # Compute loss on validation set.
                n_step = 0
                compute_summary_batch = val_size // (config['batch_size'] * 2)
                while True:
                    try:
                        n_step += 1

                        # Fetch validation samples batch.
                        length_batch, length_lab_batch, target_audio_batch, embeddings_batch, \
                            sample_path_batch, labels_batch, video_batch, mask_batch = sess.run(next_val_batch)
                        max_seq_len = max(length_batch)

                        if n_step == 1:
                            loss, summaries = sess.run(fetches=[model.loss_func, model.summaries],
                                                       feed_dict={
                                                        sequence_lengths_ph: length_batch,
                                                        target_sources_ph: target_audio_batch,
                                                        video_features_ph: video_batch,
                                                        embeddings_ph: embeddings_batch,
                                                        masks_ph: mask_batch,
                                                        audio_feat_mean_ph: audio_feat_mean,
                                                        audio_feat_std_ph: audio_feat_std,
                                                        dropout_rate_ph: 0.0
                                                       })
                        else:
                            loss = sess.run(fetches=model.loss_func,
                                            feed_dict={
                                              sequence_lengths_ph: length_batch,
                                              target_sources_ph: target_audio_batch,
                                              video_features_ph: video_batch,
                                              embeddings_ph: embeddings_batch,
                                              masks_ph: mask_batch,
                                              audio_feat_mean_ph: audio_feat_mean,
                                              audio_feat_std_ph: audio_feat_std,
                                              dropout_rate_ph: 0.0
                                            })
                        
                        if n_step == 1:
                            nframe_sum = np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']
                            val_avg_loss = loss
                        else:
                            prev_nframe_sum = nframe_sum
                            nframe_sum += np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']
                            val_avg_loss = (val_avg_loss * prev_nframe_sum + loss * np.count_nonzero(mask_batch == 0) // config['audio_feat_dim']) / nframe_sum
                

                        if n_step % 200 == 0 or n_step == 1:
                            print('Step[{:7d}] Loss[{:3.5f}]'.format(n_step, val_avg_loss))
                    except tf.errors.OutOfRangeError:
                        print('done.')
                        print('Validation loss: {:3.5f}. Best loss so far {:2.5f} [Epoch {:d} (step {:d})]'. \
                                format(val_avg_loss, best_val_loss, best_val_checkpoint[0], best_val_checkpoint[1]))
                    
                        # Save checkpoint if the current model is the best on validation set
                        if best_val_checkpoint == (0, 0) or val_avg_loss < best_val_loss:
                            checkpoint_path = os.path.join(checkpoints_dir, 'sinet')
                            save_path = saver.save(sess, checkpoint_path)
                            print('Model saved in file %s' % save_path)
                            best_val_checkpoint = (epoch_counter, tot_step)
                            best_val_loss = val_avg_loss
                            cneg_epochs = 0
                        else:
                            cneg_epochs += 1

                        break
            
                # Write Tensorboard summaries.
                tb_summary = tf.Summary()
                tb_summary.value.add(tag='Training loss full', simple_value=train_avg_loss)
                tb_summary.value.add(tag='Training loss', simple_value=train_avg_loss_fn)
                tb_summary.value.add(tag='Validation loss', simple_value=val_avg_loss)
                tb_writer.add_summary(tb_summary, epoch_counter)
                tb_writer.add_summary(summaries, epoch_counter)
                tb_writer.flush()
                
                print('')

                # Write results on log file.
                training_log_file.write('{:d}\t{:.6f}\t{:.6f}|{:.6f}\t{:.6f}\t[{:.2f}]\n' \
                   .format(epoch_counter, lr, train_avg_loss, train_avg_loss_fn, val_avg_loss, epoch_duration))
                training_log_file.flush()

                if (cneg_epochs >= config['n_earlystop_epochs']):
                    break

        if (cneg_epochs >= config['n_earlystop_epochs']):
            print('+---- Done training: early stopped ----+')
        else:
            print('+---- Done training: epoch limit reached ----+')
        print('Total training time: {:.2f} s'.format(time() - train_start_time))
        print('{:d} epochs, {:d} steps.'.format(epoch_counter, tot_step))
        print('Best validation checkpoint: {:d} ({:d}) - Loss: {:.5f}'.format(best_val_checkpoint[0], best_val_checkpoint[1], best_val_loss))


if __name__ == '__main__':
    config_file = 'C:\\Users\\Giovanni Morrone\\Documents\\Dottorato di Ricerca\\Internship AAU\\code\\scripts\\Inpainting\\configurations\\blstm_grid.config'

    train(config_file)