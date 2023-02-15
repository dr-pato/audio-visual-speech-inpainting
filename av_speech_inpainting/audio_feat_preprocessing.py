import os
from os.path import join
from glob import glob
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import math
from audio_processing import downsampling, preemphasis, get_stft, get_spectrogram, get_log_mel_spectrogram, get_mfcc, add_delta_features

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')

# Avoid printing TF log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TF constants definition
TF_INTRA_OP_PT = int(os.getenv('TF_INTRA_OP', 0))
TF_INTER_OP_PT = int(os.getenv('TF_INTER_OP', 0))


def compute_mean_std_features(audio_folder, file_prefix, out_prefix, type='spec', sample_rate=16e3, n_fft=512, window_size=25,
                              step_size=10, preemph=0, num_mel_bins=80, num_mfcc=13, delta=0, apply_mask=False, save_feat=False, file_ext='wav'):
    audio_sample_dirs = [d for d in glob(os.path.join(audio_folder, '*')) if os.path.isdir(d)]
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))

    # Create Graph
    with tf.Graph().as_default():
        samples_ph = tf.placeholder(tf.float32)
        samples_exp = tf.expand_dims(samples_ph, axis=0)
        
        # Apply pre-emphasis (if required)
        if preemph > 0:
            samples_exp = preemphasis(samples_exp, alpha=preemph)
        # Compute spectrogram 
        num_spectrogram_bins = n_fft // 2 + 1
        stft = get_stft(samples_exp, sample_rate, window_size, step_size, n_fft)
        # Select features
        if type == 'stft':
            feat_dim = num_spectrogram_bins
            features = stft
        if type == 'spec':
            feat_dim = num_spectrogram_bins
            features = get_spectrogram(stft, log=True)
        else:
            pow_spectrogram = get_spectrogram(stft, power=2)
            fbanks = get_log_mel_spectrogram(pow_spectrogram, sample_rate, num_spectrogram_bins, num_mel_bins)
            if type == 'fbanks':
                feat_dim = num_mel_bins
                features = fbanks
            elif type == 'mfcc':
                feat_dim = num_mfcc
                features = get_mfcc(fbanks, num_mfcc)
            else:
                print('Type must be "stft", "spec", "fbanks" or "mfcc". Closing...')
                exit(1)

        # Add delta features
        if delta > 0:
            feat_dim *= (delta + 1)
            features = add_delta_features(features, n_delta=delta, N=2)
        
        features = features[0]

        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT,
                                              gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            print('Computing features...')
            frame_count = 0
            tot_frame_sum = np.zeros(feat_dim)
            tot_frame_square_sum = np.zeros(feat_dim)
            for i, audio_dir in enumerate(audio_sample_dirs):
                # Read audio file
                audio_file = os.path.join(audio_dir, file_prefix + '.' + file_ext)
                rate, samples = wavfile.read(audio_file)
                samples = downsampling(samples, rate, sample_rate)
                
                # Run op
                feat = sess.run(fetches=features, feed_dict={samples_ph: samples})
                
                if apply_mask:
                    # Read mask
                    mask = np.load(os.path.join(audio_dir, 'mask.npy'))
                    feat = feat[: len(mask), : feat_dim] # discard last frequency bins and last frame
                    # Mask features
                    feat = feat * mask
                #import matplotlib.pyplot as plt
                #plt.imshow(feat.T, origin='lower')
                #plt.show()
                # Save spectrogram if required
                if save_feat:
                    feat_file = os.path.join(audio_folder, os.path.basename(audio_dir), file_prefix + '.npy')
                    np.save(feat_file, feat)

                # Update sums
                tot_frame_sum += feat.sum(axis=0)
                tot_frame_square_sum += (feat ** 2).sum(axis=0)
                if apply_mask:
                    frame_count += int(mask[:, 0].sum())
                else:
                    frame_count += len(feat)
                
            print('done. Audio files processed:', len(audio_sample_dirs))

    print('Computing mean and standard deviation of features...')
    # Compute mean and standard deviation of features
    print('Total number of frames:', frame_count)
    feat_mean = tot_frame_sum / frame_count
    feat_std = np.sqrt(tot_frame_square_sum / frame_count - feat_mean ** 2)
    print('done.')

    print('')
    print('Features mean:')
    print(feat_mean.shape)
    print(feat_mean)
    print('Features standard deviation:')
    print(feat_std.shape)
    print(feat_std)
    
    # Save mean and standard deviation
    np.save(os.path.join(audio_folder, out_prefix + '_mean.npy'), feat_mean)
    np.save(os.path.join(audio_folder, out_prefix + '_std.npy'), feat_std)
    print('Normalization data files saved.')


def save_features(audio_folder, type='spec', sample_rate=16e3, n_fft=512, window_size=25,
                  step_size=10, preemph=0, num_mel_bins=80, num_mfcc=13, delta=0, file_ext='wav'):
    audio_sample_files = glob(os.path.join(audio_folder, '*.' + file_ext))
    
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))

    # Create Graph
    with tf.Graph().as_default():
        samples_ph = tf.placeholder(tf.float32)
        samples_exp = tf.expand_dims(samples_ph, axis=0)
        
        # Apply pre-emphasis (if required)
        if preemph > 0:
            samples_exp = preemphasis(samples_exp, alpha=preemph)
        # Compute spectrogram 
        stft = get_stft(samples_exp, sample_rate, window_size, step_size, n_fft)
        num_spectrogram_bins = n_fft // 2 + 1
        # Select features
        if type == 'stft':
            feat_dim = num_spectrogram_bins
            features = stft
        elif type == 'spec':
            feat_dim = num_spectrogram_bins
            features = get_spectrogram(stft, log=True)
        else:
            pow_spectrogram = get_spectrogram(stft, power=2)
            fbanks = get_log_mel_spectrogram(pow_spectrogram, sample_rate, num_spectrogram_bins, num_mel_bins)
            if type == 'fbanks':
                feat_dim = num_mel_bins
                features = fbanks
            elif type == 'mfcc':
                feat_dim = num_mfcc
                features = get_mfcc(fbanks, num_mfcc)
            else:
                print('Type must be "spec", "fbanks" or "mfcc". Closing...')
                exit(1)

        # Add delta features
        if delta > 0:
            feat_dim *= (delta + 1)
            features = add_delta_features(features, n_delta=delta, N=2)
        
        features = features[0]

        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False,
                                              intra_op_parallelism_threads=TF_INTRA_OP_PT,
                                              inter_op_parallelism_threads=TF_INTER_OP_PT,
                                              gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            print('Computing and saving features...')
            for i, audio_file in enumerate(audio_sample_files):
                # Read audio file
                audio_seg = AudioSegment.from_file(audio_file).set_sample_width(2)
                samples = np.array(audio_seg.get_array_of_samples())
                rate = audio_seg.frame_rate
                samples = downsampling(samples, rate, sample_rate)

                # Run op
                feat = sess.run(fetches=features, feed_dict={samples_ph: samples})

                # Save spectrogram if required
                feat_file = os.path.join(audio_folder, os.path.splitext(audio_file)[0] + '.npy')
                np.save(feat_file, feat)

            print('done. Audio files processed:', len(audio_sample_files))


if __name__ == '__main__':
    audio_dir = 'C:\\Users\\Public\\aau_data\\GRID\\test_si_dataset\\test-set-lbl'
    type = 'fbanks'
    sample_rate = 16000
    fft_size = 512
    window_size = 24
    step_size = 12
    ext = 'wav'
    file_prefix = 'target'
    out_prefix = 'fbanks_norm'
    apply_mask = False
    save_feat=False

    #save_features(audio_dir, type, sample_rate, fft_size, window_size, step_size, file_ext=ext)

    compute_mean_std_features(audio_dir, file_prefix, out_prefix, type, sample_rate, fft_size,
                              window_size, step_size, apply_mask=apply_mask, save_feat=save_feat, file_ext=ext)