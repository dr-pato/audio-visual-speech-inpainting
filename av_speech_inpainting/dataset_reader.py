import os
from glob import glob
import random
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')


class DataManager:
    """Utilities to read TFRecords"""

    def __init__(self, num_audio_samples=48000, audio_feat_size=257, video_feat_size=136, buffer_size=1000, mode='fixed'):
        self.num_audio_samples = num_audio_samples
        self.audio_feat_size = audio_feat_size
        self.video_feat_size = video_feat_size
        self.buffer_size = buffer_size
        self.mode = mode


    def get_dataset(self, file_list, shuffle=True, seed=None):
        dataset = tf.data.TFRecordDataset(file_list)
        if shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=seed)

        if self.mode == 'fixed':
            dataset = dataset.map(self.read_data_format_fixed)
        elif self.mode == 'var':
            dataset = dataset.map(self.read_data_format_var)

        return dataset


    def get_dataset_group(self, file_list=[], cycle_length=100, block_length=16, shuffle=True, seed=None):
        files = tf.data.Dataset.list_files(file_list, shuffle=shuffle, seed=seed)
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=cycle_length, block_length=block_length)

        if self.mode == 'fixed':
            dataset = dataset.map(self.read_data_format_fixed)
        elif self.mode == 'var':
            dataset = dataset.map(self.read_data_format_var)

        return dataset


    def get_iterator(self, dataset, batch_size=16, n_epochs=None, drop_remainder=False):
        dataset = dataset.repeat(n_epochs)
        
        if self.mode == 'fixed':
            batch_dataset = dataset.batch(batch_size)
        elif self.mode == 'var':
            batch_dataset = dataset.padded_batch(batch_size, padded_shapes=([], [], [None], [None], [None], [None, None], [None, None]),
                                                 drop_remainder=drop_remainder)
        
        #iterator = batch_dataset.make_initializable_iterator()
        iterator = batch_dataset.make_one_shot_iterator()
        return batch_dataset, iterator


    def read_data_format_fixed(self, sample):
        context_parsed, sequence_parsed = \
                    tf.parse_single_sequence_example(sample,
                                                     context_features={
                                                      'sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
                                                      'labels_length': tf.FixedLenFeature([], dtype=tf.int64),
                                                      'target_audio_wav': tf.FixedLenFeature([self.num_audio_samples], dtype=tf.float32),
                                                      'sample_path': tf.VarLenFeature(dtype=tf.string)
                                                      },
                                                     sequence_features={
                                                      'labels': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                                                      'video_features': tf.FixedLenSequenceFeature([self.video_feat_size], dtype=tf.float32),
                                                      'mask': tf.FixedLenSequenceFeature([self.audio_feat_size], dtype=tf.float32)
                                                      })

        return tf.to_int32(context_parsed['sequence_length']), tf.to_int32(context_parsed['labels_length']), \
            tf.to_int32(context_parsed['target_audio_wav']), context_parsed['sample_path'], sequence_parsed['labels'], \
            sequence_parsed['video_features'], sequence_parsed['mask']


    def read_data_format_var(self, sample):
        context_parsed, sequence_parsed = \
                    tf.parse_single_sequence_example(sample,
                                                     context_features={
                                                      'sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
                                                      'labels_length': tf.FixedLenFeature([], dtype=tf.int64)
                                                      },
                                                     sequence_features={
                                                      'target_audio_wav': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                                                      'sample_path': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                      'labels': tf.FixedLenSequenceFeature([], dtype=tf.float32),
                                                      'video_features': tf.FixedLenSequenceFeature([self.video_feat_size], dtype=tf.float32),
                                                      'mask': tf.FixedLenSequenceFeature([self.audio_feat_size], dtype=tf.float32)
                                                     })

        return tf.to_int32(context_parsed['sequence_length']), tf.to_int32(context_parsed['labels_length']), \
            sequence_parsed['target_audio_wav'], sequence_parsed['sample_path'], sequence_parsed['labels'], \
            sequence_parsed['video_features'], sequence_parsed['mask']


if __name__ == '__main__':
    data_path = 'C:\\Users\\Public\\aau_data\\GRID\\tfrecords\\test_si_dataset\\test-set-lbl'
    num_audio_samples = int(3 * 16000)
    audio_feat_size = 257
    video_feat_size = 136
    tfrecords_mode = 'fixed'
    batch_size = 3

    data_manager = DataManager(num_audio_samples, audio_feat_size, video_feat_size, 1000, tfrecords_mode)
    files_list = glob(os.path.join(data_path, '*.tfrecord'))
    dataset = data_manager.get_dataset(files_list, shuffle=True)
    batch_dataset, it = data_manager.get_iterator(dataset, batch_size, n_epochs=1)
    next_batch = it.get_next()

    # Start session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        # Iterator initialization
        sess.run(it.initializer)

        while True:
            # Fetch batch.
            lengths, lab_lengths, target_audio_wavs, sample_paths, labels, video_feat, masks = sess.run(next_batch)

            print(lengths)
            print(target_audio_wavs.shape)
            print(video_feat.shape)
            print(sample_paths)
            print(masks[0]) 