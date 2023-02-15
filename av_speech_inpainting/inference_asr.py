import os
from glob import glob
import numpy as np
import models_asr as net
from dataset_reader import DataManager
from config_utils import load_configfile, check_trainconfiguration
from transcription2phonemes import load_dictionary, get_phonemes_from_labels

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')

# Avoid printing tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def infer(model_path, data_path_test, audio_path, out_file_prefix, dictionary_file, apply_mask=False, norm=True, batch_size=1):
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
    
    # Load normalization data
    if norm:
        audio_feat_mean = np.load(os.path.join(model_path, 'audio_features_mean.npy'))
        audio_feat_std = np.load(os.path.join(model_path, 'audio_features_std.npy'))
    else:
        audio_feat_mean = np.zeros(config['audio_feat_dim'])
        audio_feat_std = np.ones(config['audio_feat_dim'])

    # Placeholders.
    with tf.name_scope('placeholder'):
        sequence_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        target_sources_ph = tf.placeholder(tf.float32, shape=[None, config['audio_len']], name='target_sources')
        labels_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='labels_lengths')
        video_features_ph = tf.placeholder(tf.float32, shape=[None, None, config['video_feat_dim']], name='video_features')
        masks_ph = tf.placeholder(tf.float32, shape=[None, None, config['audio_feat_dim']], name='masks')
        labels_ph = tf.placeholder(tf.float32, shape=[None, None], name='labels')
        audio_feat_mean_ph = tf.placeholder(tf.float32, shape=[len(audio_feat_mean)], name='features_mean')
        audio_feat_std_ph = tf.placeholder(tf.float32, shape=[len(audio_feat_mean)], name='features_std')
        dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate')
        
    # Graph building and definition.
    print('Building ASR inference model:')
    with tf.variable_scope('asr/' + config['model']):
        if config['model'] == 'a-blstm':
            model = net.StackedBLSTMModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph,
                                          audio_feat_mean_ph, audio_feat_std_ph, dropout_rate_ph, config, input='a', apply_mask=apply_mask)
        elif config['model'] == 'v-blstm':
            model = net.StackedBLSTMModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph,
                                          audio_feat_mean_ph, audio_feat_std_ph, dropout_rate_ph, config, input='v', apply_mask=apply_mask, video_features=video_features_ph)
        elif config['model'] == 'av-blstm':
            model = net.StackedBLSTMModel(sequence_lengths_ph, labels_lengths_ph, target_sources_ph, masks_ph, labels_ph,
                                          audio_feat_mean_ph, audio_feat_std_ph, dropout_rate_ph, config, input='av', apply_mask=apply_mask, video_features=video_features_ph)
        else:
            print('Model selection must be "a-blstm", "v-blstm", "av-blstm". Closing...')
            sys.exit(1)
        model.build_graph(var_scope=config['model'])
    print('done.')
    
    
    # Load phonemes dictionary
    ph_dict = load_dictionary(dictionary_file)

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
        saver.restore(sess, os.path.join(model_path, 'asrnet'))
        print('done.\n')

        try:
            total_samples = 0
            loss_list = []
            per_list = []
            
            print('Starting inference on dataset: {:s}'.format(data_path_test))
            while True:
                # Fetch test samples batch.
                test_length, test_lab_length, test_target_audio, test_sample_path, test_labels, \
                   test_video_features, test_mask = sess.run(next_test_batch)
                
                # Compute validation loss and enhanced sources
                loss, per, test_decoded = sess.run(fetches=[model.loss, model.per, model.decoding],
                                                                      feed_dict={
                                                                        sequence_lengths_ph: test_length,
                                                                        labels_lengths_ph: test_lab_length,
                                                                        target_sources_ph: test_target_audio,
                                                                        video_features_ph: test_video_features,
                                                                        masks_ph: test_mask,
                                                                        labels_ph: test_labels,
                                                                        audio_feat_mean_ph: audio_feat_mean,
                                                                        audio_feat_std_ph: audio_feat_std,
                                                                        dropout_rate_ph: 0.0
                                                                      })
                for decoded, sample_dir in zip(test_decoded, test_sample_path.values):
                    #sample_dir = ''.join([chr(x) for x in np.trim_zeros(sample_dir)])
                    sample_dir = sample_dir.decode()
                    
                    #os.makedirs(os.path.join(audio_path, sample_dir, 'transcriptions'), exist_ok=True)
                    decoded_pad_idx = np.where(decoded == -1)[0]
                    decoded_len = len(decoded) if len(decoded_pad_idx) == 0 else decoded_pad_idx.min()
                    decoded = decoded[: decoded_len]
                    decoded_ph = get_phonemes_from_labels(decoded, ph_dict)
                    decoded_str = ','.join(decoded_ph)
                    out_filename = os.path.join(audio_path, sample_dir, out_file_prefix + '.lbl')
                    with open(out_filename, 'w') as f:
                        f.write(decoded_str)
                    
                loss_list.append(loss)
                per_list += list(per)
                total_samples += len(test_length)
                print('Recognized {:d} utterances. Total samples processed so far {:d}.'.format(len(test_length), total_samples))
        except tf.errors.OutOfRangeError:
            print('done.')

        print('Loss: {:.5}'.format(np.mean(loss_list)))
        print('PER: {:.5}'.format(np.mean(per_list)))


if __name__ == '__main__':
    model_path = 'C:\\Users\\Public\\aau_data\\GRID\\logs\\test_si_dataset\\asr_a-blstm_exp1\\netmodel'
    data_path = 'C:\\Users\\Public\\aau_data\\GRID\\tfrecords\\test_si_dataset\\test-set-lbl'
    audio_path = 'C:\\Users\\Public\\aau_data\\GRID\\test_si_dataset\\test-set-lbl'
    out_file_prefix = 'masked'
    norm = True
    dict_file = 'C:\\Users\\Public\\aau_data\\GRID\\dictionary.txt'
    batch_size = 10
    apply_mask=True

    infer(model_path, data_path, audio_path, out_file_prefix, dict_file, apply_mask, norm, batch_size)