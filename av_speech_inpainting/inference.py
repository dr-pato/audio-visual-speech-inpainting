import sys
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
import models as net
from dataset_reader import DataManager
from config_utils import load_configfile, check_trainconfiguration
import lws

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')

# Avoid printing tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def infer(model_path, data_path_test, audio_path, out_file_prefix, norm=True, oracle_phase=False, batch_size=1):
    config = load_configfile(os.path.join(model_path, 'config.txt'))
    config = check_trainconfiguration(config)

    # Create the DataManager that reads TFRecords.
    with tf.name_scope('test_batch'):
        test_data_manager = DataManager(num_audio_samples=config['audio_len'], audio_feat_size=config['audio_feat_dim'],
                                        video_feat_size=config['video_feat_dim'], buffer_size=4000, mode='fixed')
        test_files_list = glob(os.path.join(data_path_test, '*.tfrecord'))
        test_dataset = test_data_manager.get_dataset(test_files_list, shuffle=False)
        test_batch_dataset, test_it = test_data_manager.get_iterator(test_dataset, batch_size=batch_size,
                                                                     n_epochs=1, drop_remainder=False)
        next_test_batch = test_it.get_next()
    
    # Placeholders.
    with tf.name_scope('placeholder'):
        sequence_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        target_sources_ph = tf.placeholder(tf.float32, shape=[None, config['audio_len']], name='target_sources')
        video_features_ph = tf.placeholder(tf.float32, shape=[None, None, config['video_feat_dim']], name='video_features')
        masks_ph = tf.placeholder(tf.float32, shape=[None, None, config['audio_feat_dim']], name='masks')
        audio_feat_mean_ph = tf.placeholder(tf.float32, shape=[config['audio_feat_dim']], name='features_mean')
        audio_feat_std_ph = tf.placeholder(tf.float32, shape=[config['audio_feat_dim']], name='features_std')
        dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate')
        
    # Graph building and definition.
    print('Building speech inpainting inference model:')
    #if config['model'] == 'a-blstm':
    #    model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
    #                                  audio_feat_std_ph, dropout_rate_ph, config, input='a')
    #elif config['model'] == 'v-blstm':
    #    model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
    #                                  audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
    #elif config['model'] == 'av-blstm':
    #    model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
    #                                  audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
    #elif config['model'] == 'unet':
    #    model = net.UNetFConvModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
    #                               audio_feat_std_ph, dropout_rate_ph, config)
    #else:
    #    print('Model selection must be "a-blstm", "v-blstm", "av-blstm" or "unet". Closing...')
    #    sys.exit(1)
    #model.build_graph(var_scope='model')

    if config['model'] == 'av-blstm-twosteps':
        model = net.StackedBLSTM2StepsModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                            audio_feat_std_ph, dropout_rate_ph, config, video_features=video_features_ph)
        model.build_graph(var_scope=config['model'])
    else:
        with tf.variable_scope(config['model']):
            if config['model'] == 'a-blstm':
                model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                              audio_feat_std_ph, dropout_rate_ph, config, input='a')
            elif config['model'] == 'v-blstm':
                model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                              audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
                model.build_graph(var_scope=config['model'])
            elif config['model'] == 'av-blstm':
                model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                              audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
            elif config['model'] == 'unet':
                model = net.UNetFConvModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                           audio_feat_std_ph, dropout_rate_ph, config)
            else:
                print('Model selection must be "a-blstm", "v-blstm", "av-blstm", "av-blstm-twosteps" or "unet". Closing...')
                sys.exit(1)
            model.build_graph(var_scope=config['model'])
    print('Model building done.')
    print('done.\n')
    
    # Load normalization data
    if norm:
        audio_feat_mean = np.load(os.path.join(model_path, 'audio_features_mean.npy'))
        audio_feat_std = np.load(os.path.join(model_path, 'audio_features_std.npy'))
    else:
        audio_feat_mean = np.zeros(config['audio_feat_dim'])
        audio_feat_std = np.ones(config['audio_feat_dim'])

    # The inizializer operation.
    init_op = tf.group(test_it.initializer)
        
    # Start session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options = tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(init_op)

        # Load model weigths
        print('Restore weigths:')
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_path, 'sinet'))
        print('done.\n')

        # Get enhanced sources tensor op
        if oracle_phase:
            enhanced_sources_tensor = model.enhanced_sources_oracle_phase
        else:
            enhanced_sources_tensor = model.enhanced_sources

        # LWS module initialization
        lws_processor = lws.lws(384, 192, fftsize=512, mode='speech')

        try:
            total_samples = 0
            loss_hole_list = []
            
            print('Starting inference on dataset: {:s}'.format(data_path_test))
            while True:
                # Fetch test samples batch.
                test_length, test_target_audio, test_sample_path, test_video_features, test_mask = sess.run(next_test_batch)
                
                # Compute validation loss and enhanced sources
                test_enhanced_audio, loss_hole = sess.run(fetches=[enhanced_sources_tensor, model.loss],
                                                                      feed_dict={
                                                                        sequence_lengths_ph:test_length,
                                                                        target_sources_ph: test_target_audio,
                                                                        video_features_ph: test_video_features,
                                                                        masks_ph: test_mask,
                                                                        audio_feat_mean_ph: audio_feat_mean,
                                                                        audio_feat_std_ph: audio_feat_std,
                                                                        dropout_rate_ph: 0.0
                                                                      })
                for enhanced, sample_dir, mask, seq_len in zip(test_enhanced_audio, test_sample_path.values, test_mask, test_length):
                    # Reconstruct phase with LWS algorithm if required
                    if not oracle_phase:
                        stft = lws_processor.stft(enhanced)
                        mask_adj = np.zeros_like(stft)
                        mask_adj[: mask.shape[0], :mask.shape[1]] = mask
                        mag_spec = np.abs(stft)
                        ang_spec = np.angle(stft) * mask_adj
                        rec_stft = lws_processor.run_lws(mag_spec * np.exp(1j * ang_spec))
                        rec_mag = np.abs(rec_stft)
                        rec_ang = np.angle(rec_stft)
                        rec_ang_adj = ang_spec + rec_ang * (1 - mask_adj)
                        rec_stft_adj = rec_mag * np.exp(1j * rec_ang_adj)
                        enhanced = lws_processor.istft(rec_stft_adj)

                    #sample_dir = ''.join([chr(x) for x in np.trim_zeros(sample_dir)])
                    sample_dir = sample_dir.decode()
                    
                    os.makedirs(os.path.join(audio_path, sample_dir, 'enhanced'), exist_ok=True)
                    num_wav_samples = seq_len * 192
                    out_filename = os.path.join(audio_path, sample_dir, 'enhanced', out_file_prefix + '.wav')
                    wavfile.write(out_filename, 16000, enhanced[: num_wav_samples].astype(np.int16))

                loss_hole_list.append(loss_hole)
                total_samples += len(test_length)
                print('Written {:d} enhanced wavs. Total samples written so far {:d}.'.format(len(test_length), total_samples))
        except tf.errors.OutOfRangeError:
            print('done.')

        print('Loss hole: {:.5}'.format(np.mean(loss_hole_list)))
        

if __name__ == '__main__':
    model_path = 'C:\\Users\\Public\\aau_data\\LibriSpeech\\logs\\test_si_dataset\\basic_blstm\\netmodel'
    data_path = 'C:\\Users\\Public\\aau_data\\LibriSpeech\\tfrecords\\test_si_dataset\\training-set'
    audio_path = 'C:\\Users\\Public\\aau_data\\LibriSpeech\\test_si_dataset\\training-set'
    out_file_prefix = 'basic_blstm_ter'
    norm = True
    oracle_phase = False
    batch_size = 10

    infer(model_path, data_path, audio_path, out_file_prefix, norm, oracle_phase, batch_size)