import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from dataset_reader import DataManager
from audio_processing import get_stft, get_spectrogram, get_sources
import lws

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')

# Avoid printing tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mask_app(data_path, audio_path, tfrecord_mode='fixed', oracle_phase=True, audio_feat_dim=257, video_feat_dim=136, num_audio_samples=48000, batch_size=1):
    graph = tf.get_default_graph()

    # Create the DataManager that reads TFRecords.
    with tf.name_scope('test_batch'):
        data_manager = DataManager(num_audio_samples=num_audio_samples, audio_feat_size=audio_feat_dim,
                                   video_feat_size=video_feat_dim, buffer_size=4000, mode='fixed')
        files_list = glob(os.path.join(data_path, '*.tfrecord'))
        dataset = data_manager.get_dataset(files_list, shuffle=False)
        batch_dataset, it = data_manager.get_iterator(dataset, batch_size=batch_size,
                                                      n_epochs=1, drop_remainder=False)
        next_batch = it.get_next()
    
    # Placeholders.
    with tf.name_scope('placeholder'):
        sequence_lengths_ph = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        target_sources_ph = tf.placeholder(tf.float32, shape=[None, num_audio_samples], name='target_sources')
        masks_ph = tf.placeholder(tf.float32, shape=[None, None, audio_feat_dim], name='masks')

    audio_feat_mean = np.load('/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/training-set/spec_norm_mean.npy')
    audio_feat_std = np.load('/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/training-set/spec_norm_std.npy')

    # Ops definition
    target_stft = get_stft(target_sources_ph, window_size=24, step_size=12, n_fft=512, out_shape=tf.shape(masks_ph))
    masked_stft = target_stft * tf.cast(masks_ph, dtype=tf.complex64)
    masked_mag_specs = tf.abs(masked_stft)
    rec_phase = tf.angle(target_stft) if oracle_phase else tf.angle(masked_stft)
    masked_sources = get_sources(masked_mag_specs, rec_phase, num_samples=num_audio_samples)

    # Compute L1 loss
    target_spec = get_spectrogram(target_stft, log=True)
    target_spec_norm = (target_spec - audio_feat_mean) / audio_feat_std
    loss_hole_tensor = tf.reduce_sum(tf.abs(target_spec_norm) * (1 - masks_ph)) / tf.reduce_sum(1 - masks_ph)
    
    # The inizializer operation.
    init_op = tf.group(it.initializer)
        
    # Start session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(init_op)
        # LWS module initialization
        lws_processor = lws.lws(256, 128, mode='speech')

        print('Mask application on dataset: {:s}'.format(data_path))
        total_wavs = 0
        loss_hole_list = []
        try:
            while True:
                # Fetch test samples batch.
                seq_length, lab_length, target_audio, sample_path, labels, video_features, mask = sess.run(next_batch)
                
                # Compute oracle masks and enhanced sources
                masked_audio, loss_hole = sess.run(fetches=[masked_sources, loss_hole_tensor],
                                        feed_dict={
                                           sequence_lengths_ph: seq_length,
                                           target_sources_ph: target_audio,
                                           masks_ph: mask
                                           })

                # Save enhanced audio samples and optionally T-F masks
                for masked, sample_dir, seq_len in zip(masked_audio, sample_path.values, seq_length):
                    # Reconstruct phase with LWS algorithm if required
                    #if not oracle_phase:
                    #    stft = lws_processor.stft(masked)
                    #    mag_spec = np.abs(stft)
                    #    rec_stft = lws_processor.run_lws(mag_spec)
                    #    masked = lws_processor.istft(rec_stft)

                    if tfrecord_mode == 'fixed':
                        sample_dir = sample_dir.decode()
                    else:
                        sample_dir = ''.join([chr(x) for x in np.trim_zeros(sample_dir)])

                    num_wav_samples = seq_len * 192
                    masked_filename = os.path.join(audio_path, sample_dir, 'masked.wav')
                    wavfile.write(masked_filename, 16000, masked[: num_wav_samples].astype(np.int16))

                total_wavs += len(seq_length)
                loss_hole_list.append(loss_hole)
                print('Written {:d} masked wavs. Total wavs written so far {:d}.'.format(len(seq_length), total_wavs))
        except tf.errors.OutOfRangeError:
            print('done.')

        print('Loss hole: {:.5}'.format(np.mean(loss_hole_list)))

if __name__ == '__main__':
    data_path = 'C:\\Users\\Public\\aau_data\\LibriSpeech\\tfrecords\\test_si_dataset\\training-set'
    audio_path = 'C:\\Users\\Public\\aau_data\\LibriSpeech\\test_si_dataset\\training-set'
    tfrecord_mode = 'fixed'
    oracle_phase = True
    audio_feat_dim = 128
    num_audio_samples = 16384
    batch_size = 32


    mask_app(data_path, audio_path, tfrecord_mode, oracle_phase, audio_feat_dim, num_audio_samples, batch_size)