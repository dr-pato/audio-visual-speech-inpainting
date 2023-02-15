import math
from audio_processing import get_stft, get_spectrogram, get_log_mel_spectrogram, get_oracle_iam, get_sources

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')


class StackedBLSTMModel(object):
    """
    Speech recognition BLSTM model
    Input: log-compressed linear spectrogram.
    Model: stacked BLSTM.
    Output: phonemes list.
    Loss: CTC
    """
    def __init__(self, sequence_lengths, labels_lengths, target_sources, masks, labels, audio_feat_mean, audio_feat_std, dropout_rate, config,
                 audio_features=None, video_features=None, input='a', apply_mask=False, is_training=True):
        self.sequence_lengths = sequence_lengths
        self.labels_lengths = labels_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.video_features = video_features
        self.masks = masks
        self.labels = labels
        self.sparse_labels = tf.cast(tf.contrib.keras.backend.ctc_label_dense_to_sparse(labels, labels_lengths), dtype=tf.int32)
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std
        stft_shape = tf.convert_to_tensor([tf.shape(target_sources)[0], tf.reduce_max(sequence_lengths), self.audio_feat_dim])
        self.target_stft = get_stft(target_sources, window_size=24, step_size=12, n_fft=512, out_shape=stft_shape)
        self.target_spec = get_spectrogram(self.target_stft, power=2)
        if apply_mask:
            self.target_spec = self.target_spec * masks
        self.target_fbanks = get_log_mel_spectrogram(self.target_spec)
        self.target_fbanks_norm = (self.target_fbanks - audio_feat_mean) / audio_feat_std # standard normalization
        #if audio_features is None:
        #    self.audio_features = self.target_spec_norm * masks
        #else:
        #    self.audio_features = audio_features
        self.audio_features = self.target_fbanks_norm
        # input selection
        self.input_type = input
        if self.input_type == 'a':
            self.net_inputs = self.audio_features
        elif self.input_type == 'v':
            self.net_inputs = self.video_features
        elif self.input_type == 'av':
            self.net_inputs = tf.concat([self.audio_features, self.video_features], axis=2)
        self.dropout_rate = dropout_rate
        self.num_classes = config['num_asr_labels']
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        #if is_training:
        #    self.batch_size = config['batch_size']
        #else:
        #    self.batch_size = tf.shape(target_sources)[0]
        self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._inference = None
        self._loss = None
        self._train_op = None
        self._decoding = None
        self._per = None
        self._global_step = None
        self._all_vars = None
        self._train_vars = None
        self.func_loss = None
        self.reg_loss = None
        self.var_scope = None

    def build_graph(self, var_scope=''):
        """
        Build the Graph of the model.
        """
        self.inference
        self.decoding
        self.loss
        self.train_op
        self.per
        self.var_scope = var_scope
        self.all_vars
        self.train_vars
        
    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)

            if self.is_training:
                stacked_blstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                    num_layers=self.num_layers,
                                                    num_units=self.net_dim[0],
                                                    input_mode='linear_input',
                                                    direction='bidirectional',
                                                    dropout=0.0,
                                                    seed=0)
                t_net_inputs = tf.transpose(self.net_inputs, [1, 0, 2])
                t_rnn_outputs, _ = stacked_blstm(inputs=t_net_inputs)
                rnn_outputs = tf.transpose(t_rnn_outputs, [1, 0, 2])
            else:
                with tf.variable_scope('cudnn_lstm'):
                    blstm_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.net_dim[0])
                    forward_cells = [blstm_cell() for i in range(self.num_layers)]
                    backward_cells = [blstm_cell() for _ in range(self.num_layers)]
                    # Leave the scope arg unset.
                    rnn_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                         cells_fw=forward_cells,
                         cells_bw=backward_cells,
                         inputs=self.net_inputs,
                         dtype=tf.float32)

            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)
            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.num_classes], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
                logits = tf.reshape(logits, [self.batch_size, -1, self.num_classes])
            self._inference = tf.identity(logits, name='inference')

        return self._inference

    @property
    def decoding(self):
        if self._decoding is None:
            tm_logits = tf.transpose(self.inference, (1, 0, 2)) # CTC operations are defined as time major
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(tm_logits, self.sequence_lengths)
            self.sparse_decoding = tf.cast(decoded[0], tf.int32, name='sparse_decoding')
            self._decoding = tf.sparse.to_dense(self.sparse_decoding, default_value=-1, name='decoding')
            
        return self._decoding

    @property
    def loss(self):
        if self._loss is None:
            # CTC loss
            tm_logits = tf.transpose(self.inference, (1, 0, 2)) # CTC operations are defined as time major
            self.ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(self.sparse_labels, tm_logits, sequence_length=self.sequence_lengths,
                                                          preprocess_collapse_repeated=False, ctc_merge_repeated=True, 
                                                          time_major=True))
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.ctc_loss, self.regularization * self.reg_loss, name='loss')

        return self._loss

    @property
    def train_op(self):
        if self._train_op is None:
            with tf.name_scope('optimizer'):
                self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                                self.updating_step, self.learning_decay, staircase=True)
                if self.optimizer_choice == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.starter_learning_rate)
                else:
                    if self.optimizer_choice == 'sgd':
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.optimizer_choice == 'momentum':
                        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)  
                    else:
                        print('Optimizer must be either sgd, momentum or adam. Closing...')
                        sys.exit(1)
                
                self._train_op = optimizer.minimize(self.loss, global_step=self.global_step, var_list=self.train_vars, name='train_op')
        return self._train_op

    @property
    def per(self):
        if self._per is None:
            self._per = tf.edit_distance(self.sparse_decoding, self.sparse_labels, name='per')
        
        return self._per

    @property
    def global_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0, trainable=False)
        return self._global_step

    @property
    def all_vars(self):
        if self._all_vars is None:
            self._all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.var_scope)
        return self._all_vars

    @property
    def train_vars(self):
        if self._train_vars is None:
            self._train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.var_scope)
        return self._train_vars
