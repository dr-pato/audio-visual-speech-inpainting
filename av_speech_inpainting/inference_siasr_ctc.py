import sys
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
import models as net
import models_asr as asrnet
from dataset_reader_emb import DataManager
from config_utils import load_configfile, check_trainconfiguration
from transcription2phonemes import load_dictionary, get_phonemes_from_labels
import lws

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')

# Avoid printing tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def infer(model_path, model_path_asr, data_path_test, audio_path, out_file_prefix, dictionary_file, norm=True, oracle_phase=False, batch_size=1):
    config = load_configfile(os.path.join(model_path, 'config.txt'))
    config = check_trainconfiguration(config)
    config_asr = load_configfile(os.path.join(model_path_asr, 'config.txt'))
    config_asr = check_trainconfiguration(config_asr)

    # Create the DataManager that reads TFRecords.
    with tf.name_scope('test_batch'):
        test_data_manager = DataManager(num_audio_samples=config['audio_len'], audio_feat_size=config['audio_feat_dim'],
                                        video_feat_size=config['video_feat_dim'], buffer_size=4000, mode='fixed')
        test_files_list = glob(os.path.join(data_path_test, '*.tfrecord'))
        test_dataset = test_data_manager.get_dataset(test_files_list, shuffle=False)
        test_batch_dataset, test_it = test_data_manager.get_iterator(test_dataset, batch_size=batch_size,
                                                                     n_epochs=1, drop_remainder=False)
        next_test_batch = test_it.get_next()
    
    # Load normalization data
    if norm:
        audio_feat_mean = np.load(os.path.join(model_path, 'audio_features_mean.npy'))
        audio_feat_std = np.load(os.path.join(model_path, 'audio_features_std.npy'))
    else:
        audio_feat_mean = np.zeros(config['audio_feat_dim'])
        audio_feat_std = np.ones(config['audio_feat_dim'])
    audio_feat_mean_asr = np.load(os.path.join(model_path_asr, 'audio_features_mean.npy'))
    audio_feat_std_asr = np.load(os.path.join(model_path_asr, 'audio_features_std.npy'))

    # Placeholders.
    with tf.name_scope('placeholder'):
        sequence_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        labels_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='labels_lengths')
        target_sources_ph = tf.placeholder(tf.float32, shape=[None, config['audio_len']], name='target_sources')
        video_features_ph = tf.placeholder(tf.float32, shape=[None, None, config['video_feat_dim']], name='video_features')
        embeddings_ph = tf.placeholder(tf.float32, shape=[None, 512], name='embeddings')
        masks_ph = tf.placeholder(tf.float32, shape=[None, None, config['audio_feat_dim']], name='masks')
        labels_ph = tf.placeholder(tf.float32, shape=[None, None], name='labels')
        audio_feat_mean_ph = tf.placeholder(tf.float32, shape=[len(audio_feat_mean)], name='features_mean')
        audio_feat_std_ph = tf.placeholder(tf.float32, shape=[len(audio_feat_mean)], name='features_std')
        audio_feat_mean_asr_ph = tf.placeholder(tf.float32, shape=[len(audio_feat_mean_asr)], name='features_mean_asr')
        audio_feat_std_asr_ph = tf.placeholder(tf.float32, shape=[len(audio_feat_mean_asr)], name='features_std_asr')
        dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate')
        

    # Graph building and definition.
    print('Building speech inpainting inference model..')
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
            elif config['model'] == 'av-blstm':
                model = net.StackedBLSTMModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                              audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
            elif config['model'] == 'a-blstm-ssnn':
                model = net.StackedBLSTMSSNNModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                      audio_feat_std_ph, dropout_rate_ph, config, input='a')
            elif config['model'] == 'v-blstm-ssnn':
                model = net.StackedBLSTMSSNNModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                                  audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
                model.build_graph(var_scope=config['model'])
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
            elif config['model'] == 'a-blstm-ctc':
                    model = net.StackedBLSTMCTCLossModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph, audio_feat_mean_ph,
                                                         audio_feat_std_ph, dropout_rate_ph, config, input='a')
            elif config['model'] == 'v-blstm-ctc':
                model = net.StackedBLSTMCTCLossModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph, audio_feat_mean_ph,
                                                     audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
            elif config['model'] == 'av-blstm-ctc':
                model = net.StackedBLSTMCTCLossModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph, audio_feat_mean_ph,
                                                     audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
            elif config['model'] == 'a-blstm-ssnn-ctc':
                model = net.StackedBLSTMSSNNCTCLossModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph, audio_feat_mean_ph,
                                                         audio_feat_std_ph, dropout_rate_ph, config, input='a')
            elif config['model'] == 'v-blstm-ssnn-ctc':
                model = net.StackedBLSTMSSNNCTCLossModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph, audio_feat_mean_ph,
                                                         audio_feat_std_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
            elif config['model'] == 'av-blstm-ssnn-ctc':
                model = net.StackedBLSTMSSNNCTCLossModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph, audio_feat_mean_ph,
                                                         audio_feat_std_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
            elif config['model'] == 'unet':
                model = net.UNetFConvModel(sequence_lengths_ph, target_sources_ph, masks_ph, audio_feat_mean_ph,
                                           audio_feat_std_ph, dropout_rate_ph, config)
            else:
                print('Model selection must be "a-blstm", "v-blstm", "av-blstm", "av-blstm-twosteps" or "unet". Closing...')
                sys.exit(1)
            model.build_graph(var_scope=config['model'])
    print('done.')
    print('Building ASR inference model:')
    with tf.variable_scope('asr/' + config_asr['model']):
        if config_asr['model'] == 'a-blstm':
            model_asr = asrnet.StackedBLSTMModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph,
                                                 audio_feat_mean_asr_ph, audio_feat_std_asr_ph, dropout_rate_ph, config_asr, input='a')
        elif config_asr['model'] == 'v-blstm':
            model_asr = asrnet.StackedBLSTMModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph,
                                                 audio_feat_mean_asr_ph, audio_feat_std_asr_ph, dropout_rate_ph, config, input='v', video_features=video_features_ph)
        elif config_asr['model'] == 'av-blstm':
            model_asr = asrnet.StackedBLSTMModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph,
                                                 audio_feat_mean_asr_ph, audio_feat_std_asr_ph, dropout_rate_ph, config, input='av', video_features=video_features_ph)
        else:
            print('Model selection must be "a-blstm", "v-blstm", "av-blstm". Closing...')
            sys.exit(1)
        model_asr.build_graph(var_scope='asr/' + config_asr['model'])
        # Work-around bug TF
        model_asr_train_vars = []
        for v in model_asr.train_vars:
            if 'asr/' + config_asr['model'] in v.name:
                model_asr_train_vars.append(v)
        print('done.')

    # Load phonemes dictionary
    ph_dict = load_dictionary(dictionary_file)

    # The inizializer operation.
    init_op = tf.group(test_it.initializer, tf.global_variables_initializer(), tf.local_variables_initializer())
        
    # Start session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(init_op)

        # Load model weigths
        print('Restore weigths:')
        saver = tf.train.Saver(var_list=model.train_vars)
        saver.restore(sess, os.path.join(model_path, 'sinet'))
        saver_asr = tf.train.Saver(var_list=model_asr_train_vars)
        saver_asr.restore(sess, os.path.join(model_path_asr, 'asrnet'))
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
            loss_asr_list = []
            per_list = []
            
            print('Starting inference on dataset: {:s}'.format(data_path_test))
            while True:
                # Fetch test samples batch.
                test_length, test_lab_length, test_target_audio, test_embeddings, test_sample_path, test_labels, \
                    test_video_features, test_mask = sess.run(next_test_batch)
                
                # Speech inpainting inference
                test_enhanced_audio, loss_hole = \
                    sess.run(fetches=[enhanced_sources_tensor, model.loss_hole],
                             feed_dict={
                               sequence_lengths_ph:test_length,
                               labels_lengths_ph: test_lab_length,
                               target_sources_ph: test_target_audio,
                               video_features_ph: test_video_features,
                               embeddings_ph: test_embeddings,
                               masks_ph: test_mask,
                               labels_ph: test_labels,
                               audio_feat_mean_ph: audio_feat_mean,
                               audio_feat_std_ph: audio_feat_std,
                               dropout_rate_ph: 0.0
                             })

                # ASR inference
                test_decoded, loss_asr, per = \
                    sess.run(fetches=[model_asr.decoding, model_asr.loss, model_asr.per],
                             feed_dict={
                               sequence_lengths_ph:test_length,
                               labels_lengths_ph: test_lab_length,
                               target_sources_ph: test_enhanced_audio,
                               video_features_ph: test_video_features,
                               masks_ph: test_mask,
                               labels_ph: test_labels,
                               audio_feat_mean_asr_ph: audio_feat_mean_asr,
                               audio_feat_std_asr_ph: audio_feat_std_asr,
                               dropout_rate_ph: 0.0
                             })
                
                for enhanced, sample_dir, mask, seq_len, decoded in zip(test_enhanced_audio, test_sample_path.values, test_mask, test_length, test_decoded):
                    # Reconstruct phase with LWS algorithm if required
                    if not oracle_phase:
                        stft = lws_processor.stft(enhanced)
                        mask_adj = np.zeros_like(stft)
                        mask_adj[: mask.shape[0], :mask.shape[1]] = mask
                        mag_spec = np.abs(stft)
                        ang_spec = np.angle(stft) * mask_adj
                        rec_stft = lws_processor.run_lws(mag_spec * np.exp(1j * ang_spec))
                        #rec_stft = lws_processor.run_lws(mag_spec)
                        rec_mag = np.abs(rec_stft)
                        rec_ang = np.angle(rec_stft)
                        rec_ang_adj = ang_spec + rec_ang * (1 - mask_adj)
                        rec_stft_adj = rec_mag * np.exp(1j * rec_ang_adj)
                        #rec_stft_adj = rec_stft
                        enhanced = lws_processor.istft(rec_stft_adj)

                    #sample_dir = ''.join([chr(x) for x in np.trim_zeros(sample_dir)])
                    sample_dir = sample_dir.decode()
                    # Save enhanced waveform
                    os.makedirs(os.path.join(audio_path, sample_dir, 'enhanced'), exist_ok=True)
                    num_wav_samples = seq_len * 192
                    out_filename = os.path.join(audio_path, sample_dir, 'enhanced', out_file_prefix + '.wav')
                    wavfile.write(out_filename, 16000, enhanced[: num_wav_samples].astype(np.int16))
                    # Save embeddings
                    #os.makedirs(os.path.join(audio_path, sample_dir, 'embeddings'), exist_ok=True)
                    #out_filename = os.path.join(audio_path, sample_dir, 'embeddings', out_file_prefix + '.npy')
                    #np.save(out_filename, emb)
                    #out_filename = os.path.join(audio_path, sample_dir, 'embeddings', out_file_prefix + '_ext.npy')
                    #np.save(out_filename, emb_ext)
                    # Save transcription
                    decoded_pad_idx = np.where(decoded == -1)[0]
                    decoded_len = len(decoded) if len(decoded_pad_idx) == 0 else decoded_pad_idx.min()
                    decoded = decoded[: decoded_len]
                    decoded_ph = get_phonemes_from_labels(decoded, ph_dict)
                    decoded_str = ','.join(decoded_ph)
                    os.makedirs(os.path.join(audio_path, sample_dir, 'transcriptions'), exist_ok=True)
                    out_filename = os.path.join(audio_path, sample_dir, 'transcriptions', out_file_prefix + '.lbl')
                    with open(out_filename, 'w') as f:
                        f.write(decoded_str)

                loss_hole_list.append(loss_hole)
                loss_asr_list.append(loss_asr)
                per_list += list(per)
                total_samples += len(test_length)
                print('Processed {:d} utterances. Total samples processed so far {:d}.'.format(len(test_length), total_samples))
        except tf.errors.OutOfRangeError:
            print('done.')

        print('Loss hole: {:.5}'.format(np.mean(loss_hole_list)))
        print('Loss ASR: {:.5}'.format(np.mean(loss_asr_list)))
        print('PER: {:.5}'.format(np.mean(per_list)))
        

if __name__ == '__main__':
    model_path = 'C:\\Users\\Public\\aau_data\\GRID\\logs\\test_si_dataset\\a-blstm_exp0\\netmodel'
    model_path_asr = 'C:\\Users\\Public\\aau_data\\GRID\\logs\\test_si_dataset\\asr_a-blstm_exp1\\netmodel'
    data_path = 'C:\\Users\\Public\\aau_data\\GRID\\tfrecords\\test_si_dataset\\test-set-lbl'
    audio_path = 'C:\\Users\\Public\\aau_data\\GRID\\test_si_dataset\\test-set-lbl'
    out_file_prefix = 'a-blstm_exp0'
    norm = True
    oracle_phase = False
    batch_size = 10
    dict_file = 'C:\\Users\\Public\\aau_data\\GRID\\dictionary.txt'

    infer(model_path, model_path_asr, data_path, audio_path, out_file_prefix, dict_file, norm, oracle_phase, batch_size)