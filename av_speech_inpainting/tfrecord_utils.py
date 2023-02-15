from glob import glob
import os
import shutil
import random
import numpy as np
import math
from pydub import AudioSegment
from audio_processing import downsampling
from av_sync import sync_audio_visual_features
from face_landmarks import get_motion_vector
from transcription2phonemes import load_dictionary, get_labels

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')


def serialize_sample_fixed(seq_len, lab_len, target_audio_wav, video_features, mask, labels, sample_path):
    # The object we return
    example = tf.train.SequenceExample()

    # Non-sequential features of our example
    example.context.feature['sequence_length'].int64_list.value.append(seq_len)
    example.context.feature['labels_length'].int64_list.value.append(lab_len)
    example.context.feature['target_audio_wav'].float_list.value.extend(target_audio_wav)
    example.context.feature['sample_path'].bytes_list.value.append(sample_path.encode())
    
    # Feature lists for the sequential features of our example
    fl_mask = example.feature_lists.feature_list['mask']
    fl_video_features = example.feature_lists.feature_list['video_features']
    fl_labels = example.feature_lists.feature_list['labels']

    for video_feat in video_features:
        fl_video_features.feature.add().float_list.value.extend(video_feat)
    for mask_feat in mask:
        fl_mask.feature.add().float_list.value.extend(mask_feat)
    for label in labels:
        fl_labels.feature.add().float_list.value.append(label)

    return example


def serialize_sample_var(seq_len, lab_len, target_audio_wav, video_features, mask, labels, sample_path):
    # The object we return
    example = tf.train.SequenceExample()
    
    # Non-sequential features of our example
    example.context.feature['sequence_length'].int64_list.value.append(seq_len)
    example.context.feature['labels_length'].int64_list.value.append(lab_len)

    # Feature lists for the sequential features of our example
    fl_target_audio_wav = example.feature_lists.feature_list['target_audio_wav']
    fl_video_features = example.feature_lists.feature_list['video_features']
    fl_mask = example.feature_lists.feature_list['mask']
    fl_labels = example.feature_lists.feature_list['labels']
    fl_sample_path = example.feature_lists.feature_list['sample_path']

    for target_audio_el in target_audio_wav:
        fl_target_audio_wav.feature.add().float_list.value.append(target_audio_el)
    for video_feat in video_features:
        fl_video_features.feature.add().float_list.value.extend(video_feat)
    for mask_feat in mask:
        fl_target.feature.add().float_list.value.extend(mask_feat)
    for label in labels:
        fl_labels.feature.add().float_list.value.append(label)
    for sample_path_el in sample_path:
        fl_mix_audio_path.feature.add().int64_list.value.append(ord(sample_path_el))

    return example


def create_tfrecords_training(data_path, dest_dir, ph_dict, tfrecord_mode='fixed'):
    sample_dir_list =  sorted([d for d in glob(os.path.join(data_path, '*')) if os.path.isdir(d)])

    file_counter = 0
    seq_lengths = []
    for i, sample_dir in enumerate(sample_dir_list):
        print(str(i) + ' - ' + sample_dir)

        # Read audio files
        target_audio_seg = AudioSegment.from_file(os.path.join(sample_dir, 'target.wav')).set_sample_width(2)
        target_audio_wav = np.array(target_audio_seg.get_array_of_samples())
        # Read mask
        mask = np.load(os.path.join(sample_dir, 'mask.npy'))
        seq_len = len(mask)
        # Read face landmarks
        face_land = np.load(os.path.join(sample_dir, 'landmarks.npy')).reshape((-1, 136))
        # AV alignment
        video_features = sync_audio_visual_features(mask, face_land, tot_frames=75, min_frames=70)
        if video_features is None:
            print('Skipped. Video features corrupted.')
            continue
        # Compute landmark motion vector
        video_features = get_motion_vector(video_features, delta=1)
        # Read transcription
        with open(os.path.join(sample_dir, 'transcription.lbl')) as f:
            transcription = f.read()
        labels = get_labels(transcription, ph_dict)
        lab_len = len(labels)
        labels = np.pad(labels, (0, 50 - len(labels)), mode='constant') # padding at end

        # Load video features mean and standard deviation
        video_feat_mean = np.load(os.path.join(sample_dir, 'video_feat_mean.npy')).flatten()
        video_feat_std = np.load(os.path.join(sample_dir, 'video_feat_std.npy')).flatten()
        # Apply normalization
        video_features = (video_features - video_feat_mean) / video_feat_std
        
        seq_lengths.append(seq_len)
        file_counter += 1

        # Create TFRecord
        tfrecord_file = os.path.join(dest_dir, 'data_{:05d}.tfrecord'.format(file_counter))
        with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
            sample_path = os.path.basename(sample_dir)
            if tfrecord_mode == 'fixed':
                serialized_sample = serialize_sample_fixed(seq_len, lab_len, target_audio_wav, video_features, mask, labels, sample_path)
            elif tfrecord_mode == 'var':
                serialized_sample = serialize_sample_var(seq_len, lab_len, target_audio_wav, video_features, mask, labels, sample_path)
            writer.write(serialized_sample.SerializeToString())

    # Write file with sequence lengths
    np.save(os.path.join(dest_dir, 'seq_lengths.npy'), np.array(seq_lengths))
        
    return file_counter


def create_dataset(data_path, dest_dir, dictionary_file, tfrecord_mode='fixed'):
    train_data_path = os.path.join(data_path, 'training-set')
    val_data_path = os.path.join(data_path, 'validation-set')
    test_data_path = os.path.join(data_path, 'test-set')
    train_dest_dir = os.path.join(dest_dir, 'training-set')
    val_dest_dir = os.path.join(dest_dir, 'validation-set')
    test_dest_dir = os.path.join(dest_dir, 'test-set')
    
    # Create tfrecord directories if they do not exist
    for folder in (train_dest_dir, val_dest_dir, test_dest_dir):
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Load phonemes dictionary
    ph_dict = load_dictionary(dictionary_file)

    # Generate TFRecords
    print('Creating training TFRecords...')
    num_train_samples = create_tfrecords_training(train_data_path, train_dest_dir, ph_dict, tfrecord_mode)
    
    print('Creating validation TFRecords...')
    num_val_samples = create_tfrecords_training(val_data_path, val_dest_dir, ph_dict, tfrecord_mode)
    
    print('Creating test TFRecords...')
    num_test_samples = create_tfrecords_training(test_data_path, test_dest_dir, ph_dict, tfrecord_mode)

    print('')
    print('Samples successfully generated:')
    print('-> Training:', num_train_samples)
    print('-> Validation:', num_val_samples)
    print('-> Test:', num_test_samples)


def group_tfrecords(input_dir, output_dir, group_size=16, delete_input_dir=False):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Copy sequence lengths file
    shutil.copy(os.path.join(input_dir, 'seq_lengths.npy'), os.path.join(output_dir, 'seq_lengths.npy'))

    # Load sequence lengths from NPY file
    try:
        seq_lengths = np.load(os.path.join(input_dir, 'seq_lengths.npy'))
    except IOError:
        raise IOError('Cannot find seq_lengths.npy in directory', input_dir)

    tfrecord_files = sorted(glob(os.path.join(input_dir, '*.tfrecord')))
    if len(tfrecord_files) != len(seq_lengths):
        raise ValueError('Non matching number of input files [{:d}] and files reported in seq_lengths.npy [{:d}]'.format(len(tfrecord_files), len(seq_lengths)))

    print('Number of original TFRecords:', len(seq_lengths))

    os.makedirs(output_dir, exist_ok=True)
    
    # Sort TFRecord filenames by sequence lengths in seq_lengths array
    # Since mixed-audio samples with same base audio have the same length we artificially
    # modify their lengths to decrease the probability of grouping together
    rand_seq_lengths = seq_lengths + np.random.rand(len(seq_lengths)) * 10
    idx = np.argsort(rand_seq_lengths)
    tfrecord_files_ord = list(np.array(tfrecord_files)[idx])
    
    # Create the DataManager that reads TFRecords.
    filenames_ph = tf.placeholder(tf.string, shape=[None])
    data_manager = DataManager(mode='var')
    dataset = data_manager.get_dataset(file_list=filenames_ph, shuffle=False)
    dataset, iterator = data_manager.get_iterator(dataset, batch_size=1, n_epochs=1)
    next_batch = iterator.get_next()
    
    # Start session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        all_samples = 0
        n_tfrecs = 0

        for i in range(0, len(tfrecord_files), group_size):
            seq_length_list = []
            base_audio_list = []
            noise_audio_list = []
            mix_audio_list = []
            labels_list = []
            mix_audio_path_list = []
            file_list = tfrecord_files_ord[i: i + group_size]

            # Initialize iterator and feed with filenames
            sess.run(iterator.initializer, feed_dict={filenames_ph: file_list})

            while True:
                try:
                    # Read TFRecord with single sample
                    seq_length, labels, base_audio, noise_audio, mix_audio, mix_audio_path = sess.run(next_batch)

                    # Add data to lists
                    seq_length_list.append(seq_length[0])
                    base_audio_list.append(base_audio[0])
                    noise_audio_list.append(noise_audio[0])
                    mix_audio_list.append(mix_audio[0])
                    labels_list.append(labels[0])
                    mix_audio_path_list.append(''.join([chr(x) for x in mix_audio_path[0]]))
                except tf.errors.OutOfRangeError:
                    print('Done loading examples for TFRecord:', n_tfrecs)
                    break

            # Write grouped TFRecord
            tfrecord_file = os.path.join(output_dir, 'data_{:05d}.tfrecord'.format(n_tfrecs))
        
            with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
                for i in range(len(seq_length_list)):
                    audio_wavs = [base_audio_list[i], noise_audio_list[i], mix_audio_list[i]]
                    serialized_sample = serialize_sample_var(seq_length_list[i], audio_wavs, labels_list[i], mix_audio_path_list[i])
                    writer.write(serialized_sample.SerializeToString())
            
            # Delete original TFrecords if required
            if delete_input_dir:
                for file in file_list:
                    os.remove(file)

            n_tfrecs += 1
            all_samples += len(seq_length_list)
            print('Done writing tfrecord: {:s}. Number of samples: {:d}'.format(tfrecord_file, len(seq_length_list)))
            print('Number of processed samples so far:', all_samples)

    # Delete directory if required
    if delete_input_dir:
        shutil.rmtree(input_dir)

    print('')
    print('Generation of grouped TFRecords completed.')
    print('Number of output TFRecords:', n_tfrecs)
    print('Total number of processed samples:', all_samples)


if __name__ == '__main__':
    data_path = 'C:\\Users\\Public\\aau_data\\GRID\\test_si_dataset'
    dest_dir = 'C:\\Users\\Public\\aau_data\\GRID\\tfrecords\\test_si_dataset'
    dict_file = 'C:\\Users\\Public\\aau_data\\GRID\\dictionary.txt'
    tfrecord_mode = 'fixed'

    create_dataset(data_path, dest_dir, dict_file, tfrecord_mode)