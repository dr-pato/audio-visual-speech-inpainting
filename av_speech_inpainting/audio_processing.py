from scipy import signal

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')


def downsampling(samples, sample_rate, downsample_rate):
    secs = len(samples) / float(sample_rate)
    num_samples = int(downsample_rate * secs)
    
    if sample_rate != downsample_rate:
        return signal.resample(samples, num_samples)
    else:
        return samples


def preemphasis(sources, alpha=0.95):
    sources_shape = tf.shape(sources)
    return sources - alpha * tf.concat([tf.zeros([sources_shape[0], 1]),
                                        tf.slice(sources, begin=[0, 0], size=[sources_shape[0], sources_shape[1] - 1])], axis=1)


def get_stft(sources, sample_rate=16000, window_size=25, step_size=10, n_fft=512, out_shape=[0, 0, 0]):
    """Compute STFT"""
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    
    # Explicit padding at start for correct reconstruction
    #paddings = [[0, 0], [n_fft // 2 - step_frame_size // 2, 0]]
    #sources_pad = tf.pad(sources, paddings)
    
    # Compute STFTs
    stfts = tf.contrib.signal.stft(sources, fft_length=n_fft, frame_length=window_frame_size,
                                   frame_step=step_frame_size, pad_end=True)
    
    stfts = tf.cond(tf.reduce_all(tf.equal(out_shape, 0)),
                    lambda: stfts,
                    lambda: tf.slice(stfts, begin=[0, 0, 0], size=out_shape))
    
    return stfts


def get_spectrogram(stfts, power=1, log=False, out_shape=[0, 0, 0]):
    spectrograms = tf.abs(stfts)
    if power != 1:
        spectrograms = spectrograms ** power
    if log:
        spectrograms = tf.log(spectrograms + 1e-6)

    spectrograms = tf.cond(tf.reduce_all(tf.equal(out_shape, 0)),
                           lambda: spectrograms,
                           lambda: tf.slice(spectrograms, begin=[0, 0, 0], size=out_shape))

    return spectrograms


def get_log_mel_spectrogram(spectrograms, sample_rate=16000, num_spec_bins=257, num_mel_bins=80, lower_edge_freq=125, upper_edge_freq=7600, eps=1e-6, out_shape=[0, 0, 0]):
    if upper_edge_freq is None:
        upper_edge_freq = sample_rate / 2

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spec_bins, sample_rate,
                                                                        lower_edge_freq, upper_edge_freq)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, axes=1)
    log_mel_spectrograms = tf.log(mel_spectrograms + eps)

    tf.cond(tf.reduce_all(tf.equal(out_shape, 0)),
            lambda: log_mel_spectrograms,
            lambda: tf.slice(log_mel_spectrograms, begin=[0, 0, 0], size=out_shape))

    return log_mel_spectrograms


def get_mfcc(log_mel_spectrograms, num_mfccs=13, out_shape=[0, 0, 0]):
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., : num_mfccs]

    mfccs = tf.cond(tf.reduce_all(tf.equal(out_shape, 0)),
                   lambda: mfccs,
                   lambda: tf.slice(mfccs, begin=[0, 0, 0], size=out_shape))

    return mfccs


def delta(features, N=2):
    denominator = 2 * sum([i ** 2 for i in range(1, N+1)])
    delta_features_sum = tf.zeros_like(features)
    paddings = paddings = [[0, 0], [1, 1], [0, 0]]
    features_padded = features
    for i in range(1, N+1):
        features_padded = tf.pad(features_padded, paddings, mode='SYMMETRIC')
        delta_features_sum += i * (features_padded[:, i*2:, :] - features_padded[:, :-i*2, :])
    
    return delta_features_sum / denominator


def add_delta_features(features, n_delta=2, N=2):
    full_features = [features]
    cur_features = features
    for i in range(n_delta):
        cur_features = delta(cur_features, N)
        full_features.append(cur_features)
    
    return tf.concat(full_features, axis=2)


def asr_preprocessing(input_sources, type='mfcc', preemph=0.95, num_spec_bins=257, num_mel_bins=80, num_mfccs=13, n_delta=3,
                      feat_mean=None, feat_std=None, stft_shape=[0, 0, 0], name='features'):
    #preemph_sources = preemphasis(input_sources, alpha=preemph)
    #stfts = get_stft(preemph_sources, out_shape=stft_shape)
    #specs = get_spectrogram(stfts, power=2)
    #fbanks = get_log_mel_spectrogram(specs, num_spec_bins=num_spec_bins, num_mel_bins=num_mel_bins)
    #mfccs = get_mfcc(fbanks, num_mfccs=num_mfccs)
    #mfccs_delta = add_delta_features(mfccs, n_delta=n_delta)
    #if feat_mean is not None and feat_std is not None:
    #    features = (mfccs_delta - feat_mean) / feat_std

    # Apply pre-emphasis (if required)
    if preemph > 0:
        preemph_sources = preemphasis(input_sources, alpha=preemph)
    # Compute spectrogram 
    stfts = get_stft(preemph_sources, out_shape=stft_shape)
    # Select features
    if type == 'spec':
        features = get_spectrogram(stfts, power=0.3)
    else:
        pow_spectrograms = get_spectrogram(stfts, power=2)
        fbanks = get_log_mel_spectrogram(pow_spectrograms, num_spec_bins=num_spec_bins, num_mel_bins=num_mel_bins)
        if type == 'fbanks':
            features = fbanks
        elif type == 'mfcc':
            features = get_mfcc(fbanks, num_mfccs=num_mfccs)
    
    if n_delta > 0:
        features = add_delta_features(features, n_delta=n_delta)
    
    if feat_mean is None and feat_std is not None:
        feat_mean = tf.expand_dims(tf.reduce_mean(features, axis=1), axis=1)
    if feat_mean is not None and feat_std is not None:
        features = (features - feat_mean) / feat_std
    
    return tf.identity(features, name=name)


def reconstruct_sources(stfts, num_samples=0, sample_rate=16000, window_size=16, step_size=8):
    """Compute inverse STFT"""
    window_frame_size = int(round(window_size / 1e3 * sample_rate))
    step_frame_size = int(round(step_size  / 1e3 * sample_rate))
    reconstructed_sources = tf.contrib.signal.inverse_stft(stfts, frame_length=window_frame_size,
                                                           frame_step=step_frame_size,
                                                           window_fn=tf.contrib.signal.inverse_stft_window_fn(step_frame_size))

    reconstructed_sources = tf.cond(tf.greater(num_samples, 0),
                                    lambda: tf.slice(reconstructed_sources, begin=[0,0], size=[tf.shape(stfts)[0], num_samples]),
                                    lambda: reconstructed_sources)
    
    return reconstructed_sources


def get_sources(mag_spectrograms, rec_ang_spectrograms, num_samples=48000, sample_rate=16000, window_size=24, step_size=12):
    """Get waveform from magnitude and phase of STFT"""
    stfts = tf.complex(real=mag_spectrograms * tf.cos(rec_ang_spectrograms), imag=mag_spectrograms * tf.sin(rec_ang_spectrograms))
    
    return reconstruct_sources(stfts, num_samples, sample_rate, window_size, step_size)


def get_oracle_iam(target_stft, mixed_stft, clip_value=10):
    """Get oracle Ideal Amplitude Mask (IAM) from target and mixed spectrograms"""
    target_spectrograms_mag = get_spectrogram(target_stft)
    mixed_spectrograms_mag = get_spectrogram(mixed_stft)
    iam = target_spectrograms_mag / mixed_spectrograms_mag

    return tf.cast(tf.clip_by_value(iam, 0, clip_value), tf.float32)


def get_oracle_ipsm(target_stft, mixed_stft, min_clip_value=0, max_clip_value=10):
    """Get oracle Ideal Phase Sensitive Mask (IPSM) from target and mixed spectrograms"""
    target_spectrograms_mag = get_spectrogram(target_stft)
    target_spectrograms_ang = tf.angle(target_stft)
    mixed_spectrograms_mag = get_spectrogram(mixed_stft)
    mixed_spectrograms_ang = tf.angle(mixed_stft)
    ipsm = target_spectrograms_mag * tf.cos(mixed_spectrograms_ang - target_spectrograms_ang) / mixed_spectrograms_mag

    return tf.clip_by_value(ipsm, min_clip_value, max_clip_value)


