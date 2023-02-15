import math
from audio_processing import get_stft, get_spectrogram, get_oracle_iam, get_sources, add_delta_features
from unet_layers import encoder_layer_fconv, decoder_layer_fconv

try:
    import tensorflow as tf
except ImportError:
    print('Failed to import TensorFlow module.')


class StackedBLSTMModel(object):
    """
    Speech inpainting BLSTM model
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: stacked BLSTM.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
    """

    def __init__(self, sequence_lengths, target_sources, masks, audio_feat_mean, audio_feat_std, dropout_rate, config,
                 audio_features=None,video_features=None, input='a', is_training=True):
        self.sequence_lengths = sequence_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.video_features = video_features
        self.masks = masks
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std
        stft_shape = tf.convert_to_tensor([tf.shape(target_sources)[0], tf.reduce_max(sequence_lengths), self.audio_feat_dim])
        self.target_stft = get_stft(target_sources, window_size=24, step_size=12, n_fft=512, out_shape=stft_shape)
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        if audio_features is None:
            self.audio_features = self.target_spec_norm * masks
        else:
            self.audio_features = audio_features
        # input selection
        self.input_type = input
        if self.input_type == 'a':
            self.net_inputs = self.audio_features
        elif self.input_type == 'v':
            self.net_inputs = self.video_features
        elif self.input_type == 'av':
            self.net_inputs = tf.concat([self.audio_features, self.video_features], axis=2)
        self.dropout_rate = dropout_rate
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._summaries = None
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
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
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
                weights = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

        return self._inference

    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            # Restore original non-masked TF-bins
            #prediction = self.target_spec_norm * self.masks + self.inference * (1 - self.masks) 
            prediction = self.inference
            prediction = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            # L1 loss
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            # L2 loss
            #self.loss_hole = tf.identity(tf.nn.l2_loss((self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            #self.loss_valid = tf.identity(tf.nn.l2_loss((self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            #self.loss_func = tf.add(6 * self.loss_hole, self.loss_valid, name='func_loss')
            #self.loss_func = self.loss_hole
            self.loss_func = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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


class StackedBLSTM2StepsModel(object):
    """
    2-Steps speech inpainting BLSTM model
    """
    def __init__(self, sequence_lengths, target_sources, masks, audio_feat_mean, audio_feat_std, dropout_rate, config, video_features, is_training=True):
        self.sequence_lengths = sequence_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.video_features = video_features
        self.masks = masks
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std

        # Define model
        with tf.variable_scope('v-blstm'):
            self.video_model = StackedBLSTMModel(sequence_lengths, target_sources, masks, audio_feat_mean,
                                                 audio_feat_std, dropout_rate, config, video_features=video_features,
                                                 input='v', is_training=is_training)
            self.video_model.build_graph(var_scope='v-blstm')
        with tf.variable_scope('av-blstm-twosteps'):
            self.av_model = StackedBLSTMModel(sequence_lengths, target_sources, masks, audio_feat_mean,
                                              audio_feat_std, dropout_rate, config, audio_features=self.video_model.prediction,
                                              video_features=video_features, input='av', is_training=is_training)
            self.av_model.build_graph(var_scope='av-blstm-twosteps')
        
        # Define training parameters
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._summaries = None

    def build_graph(self, var_scope=''):
        self.target_spec_norm = self.av_model.target_spec_norm
        self.inference = self.av_model.inference
        self.video_prediction = self.video_model.prediction
        self.prediction = self.av_model.prediction
        self.loss = self.av_model.loss
        self.loss_func = self.av_model.loss_func
        self.train_op = self.av_model.train_op
        self.learning_rate = self.av_model.learning_rate
        self.global_step = self.av_model.global_step
        self.enhanced_sources = self.av_model.enhanced_sources
        self.enhanced_sources_oracle_phase = self.av_model.enhanced_sources_oracle_phase
        self.var_scope = var_scope
        self.all_vars = self.av_model.all_vars
        self.train_vars = self.av_model.train_vars
        self.summaries

    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_video_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.video_prediction, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Video_enhanced_spectrogram', mag_enhanced_video_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries


class UNetPConvModel(object):
    """
    Speech inpainting U-Net model with partial convolutions
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: U-Net.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
    """

    def __init__(self, sequence_lengths, target_sources, masks, feat_mean, audio_feat_std, dropout_rate, config, is_training=True):
        self.sequence_lengths = sequence_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.masks = masks
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std
        stft_shape = tf.convert_to_tensor([tf.shape(target_sources)[0], tf.reduce_max(sequence_lengths), self.audio_feat_dim])
        self.target_stft = get_stft(target_sources, window_size=16, step_size=8, n_fft=256, out_shape=stft_shape)
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        self.net_inputs = self.target_spec_norm * masks
        self.dropout_rate = dropout_rate
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.learning_rate = config['learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._summaries = None
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
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
        self.var_scope = var_scope
        self.all_vars
        self.train_vars


    @property
    def inference(self):
        if self._inference is None:
            # add features channel dim
            inputs = tf.expand_dims(self.net_inputs, axis=3)
            with tf.name_scope('down_conv'):
                e_conv1 = encoder_layer_pconv(inputs, 7, 1, 16, stride=2, batch_norm=False, is_training=self.is_training)
                e_conv2 = encoder_layer_pconv(e_conv1, 5, 16, 32, stride=2, is_training=self.is_training)
                e_conv3 = encoder_layer_pconv(e_conv2, 5, 32, 64, stride=2, is_training=self.is_training)
                e_conv4 = encoder_layer_pconv(e_conv3, 3, 64, 128, stride=2, is_training=self.is_training)
                e_conv5 = encoder_layer_pconv(e_conv4, 3, 128, 128, stride=2, is_training=self.is_training)
                e_conv6 = encoder_layer_pconv(e_conv5, 3, 128, 128, stride=2, is_training=self.is_training)
                
            with tf.name_scope('up_conv'):
                d_conv1 = decoder_layer_pconv(e_conv6, e_conv5, 3, 256, 128, is_training=self.is_training)
                d_conv2 = decoder_layer_pconv(d_conv1, e_conv4, 3, 256, 128, is_training=self.is_training)
                d_conv3 = decoder_layer_pconv(d_conv2, e_conv3, 3, 192, 64, is_training=self.is_training)
                d_conv4 = decoder_layer_pconv(d_conv3, e_conv2, 3, 96, 32, is_training=self.is_training)
                d_conv5 = decoder_layer_pconv(d_conv4, e_conv1, 3, 48, 16, is_training=self.is_training)
                d_conv6 = decoder_layer_pconv(d_conv5, inputs, 3, 17, 1, batch_norm=False, is_training=self.is_training)

            outputs = encoder_layer_fconv(d_conv6, 1, 1, 1, stride=1, batch_norm=False, activation=False, is_training=self.is_training)
            self._inference = tf.squeeze(outputs, axis=3, name='inference')

        return self._inference


    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            masked_inference = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * self.inference
            self._prediction = tf.identity(masked_inference, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (self.masks)) / tf.reduce_sum(self.masks), name='loss_valid')
            #self.loss_func = tf.add(6 * self.loss_hole, self.loss_valid, name='func_loss')
            self.loss_func = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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


class UNetFConvModel(object):
    """
    Speech inpainting U-Net model with full convolutions
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: U-Net.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
    """

    def __init__(self, sequence_lengths, target_sources, masks, audio_feat_mean, audio_feat_std, dropout_rate, config, is_training=True):
        self.sequence_lengths = sequence_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.masks = masks
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std
        stft_shape = tf.convert_to_tensor([tf.shape(target_sources)[0], tf.reduce_max(sequence_lengths), self.audio_feat_dim])
        self.target_stft = get_stft(target_sources, window_size=16, step_size=8, n_fft=256, out_shape=stft_shape)
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        self.net_inputs = self.target_spec_norm * masks
        self.dropout_rate = dropout_rate
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.learning_rate = config['learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._summaries = None
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
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
        self.var_scope = var_scope
        self.all_vars
        self.train_vars


    @property
    def inference(self):
        if self._inference is None:
            # add features channel dim
            inputs = tf.expand_dims(self.net_inputs, axis=3)
            with tf.name_scope('down_conv'):
                e_conv1 = encoder_layer_fconv(inputs, 7, 1, 16, batch_norm=False, is_training=self.is_training)
                e_conv2 = encoder_layer_fconv(e_conv1, 5, 16, 32, is_training=self.is_training)
                e_conv3 = encoder_layer_fconv(e_conv2, 5, 32, 64, is_training=self.is_training)
                e_conv4 = encoder_layer_fconv(e_conv3, 3, 64, 128, is_training=self.is_training)
                e_conv5 = encoder_layer_fconv(e_conv4, 3, 128, 128, is_training=self.is_training)
                e_conv6 = encoder_layer_fconv(e_conv5, 3, 128, 128, is_training=self.is_training)
                
            with tf.name_scope('up_conv'):
                d_conv1 = decoder_layer_fconv(e_conv6, e_conv5, 3, 256, 128, is_training=self.is_training)
                d_conv2 = decoder_layer_fconv(d_conv1, e_conv4, 3, 256, 128, is_training=self.is_training)
                d_conv3 = decoder_layer_fconv(d_conv2, e_conv3, 3, 192, 64, is_training=self.is_training)
                d_conv4 = decoder_layer_fconv(d_conv3, e_conv2, 3, 96, 32, is_training=self.is_training)
                d_conv5 = decoder_layer_fconv(d_conv4, e_conv1, 3, 48, 16, is_training=self.is_training)
                d_conv6 = decoder_layer_fconv(d_conv5, inputs, 3, 17, 1, is_training=self.is_training)

            outputs = encoder_layer_fconv(d_conv6, 1, 1, 1, stride=1, batch_norm=False, activation=False, is_training=self.is_training)
            self._inference = tf.squeeze(outputs, axis=3, name='inference')

        return self._inference


    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            inference_cut = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * self.inference
            self._prediction = tf.identity(inference_cut, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (self.masks)) / tf.reduce_sum(self.masks), name='loss_valid')
            #self.loss_func = tf.add(6 * self.loss_hole, self.loss_valid, name='func_loss')
            self.loss_func = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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


class StackedBLSTMSSNNModel(object):
    """
    Speech inpainting BLSTM model with SSNN
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: stacked BLSTM.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
    """

    def __init__(self, sequence_lengths, target_sources, masks, audio_feat_mean, audio_feat_std, dropout_rate, config,
                 audio_features=None,video_features=None, input='a', is_training=True):
        self.sequence_lengths = sequence_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.video_features = video_features
        self.masks = masks
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std
        stft_shape = tf.convert_to_tensor([tf.shape(target_sources)[0], tf.reduce_max(sequence_lengths), self.audio_feat_dim])
        self.target_stft = get_stft(target_sources, window_size=24, step_size=12, n_fft=512, out_shape=stft_shape)
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        if audio_features is None:
            self.audio_features = self.target_spec_norm * masks
        else:
            self.audio_features = audio_features
        # input selection
        self.input_type = input
        if self.input_type == 'a':
            self.net_inputs = self.audio_features
        elif self.input_type == 'v':
            self.net_inputs = self.video_features
        elif self.input_type == 'av':
            self.net_inputs = tf.concat([self.audio_features, self.video_features], axis=2)
        self.dropout_rate = dropout_rate
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.int_layer = config['integration_layer']
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._speaker_embedding = None
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._summaries = None
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
        self.speaker_embedding
        self.inference
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
        self.var_scope = var_scope
        self.all_vars
        self.train_vars
        

    @property
    def speaker_embedding(self):
        if self._speaker_embedding is None:
            with tf.variable_scope('speaker_embedding'):
                weights_1 = tf.Variable(tf.truncated_normal([self.audio_feat_dim * 2, 200], stddev=1.0 / math.sqrt(self.audio_feat_dim),
                      dtype=tf.float32), name='weights_1')
                biases_1 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_1')
                weights_2 = tf.Variable(tf.truncated_normal([200, 200], stddev=1.0 / math.sqrt(200), dtype=tf.float32), name='weights_2')
                biases_2 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_2')
                weights_3 = tf.Variable(tf.truncated_normal([200, 200], stddev=1.0 / math.sqrt(200), dtype=tf.float32), name='weights_3')
                biases_3 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_3')
                #weights_4 = tf.Variable(tf.truncated_normal([400, 200], stddev=1.0 / math.sqrt(200), dtype=tf.float32), name='weights_4')
                #biases_4 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_4')

                #prev_spec_1 = tf.pad(self.audio_features, [[0, 0], [1,0], [0,0]], mode='CONSTANT')[:, :-1, :]
                #prev_spec_2 = tf.pad(self.audio_features, [[0, 0], [2,0], [0,0]], mode='CONSTANT')[:, :-2, :]
                #flw_spec_1 = tf.pad(self.audio_features, [[0, 0], [0,1], [0,0]], mode='CONSTANT')[:, 1:, :]
                #flw_spec_2 = tf.pad(self.audio_features, [[0, 0], [0,2], [0,0]], mode='CONSTANT')[:, 2:, :]
                #inp = tf.concat([prev_spec_2, prev_spec_1, self.audio_features, flw_spec_1, flw_spec_2], axis=2)
                inp = add_delta_features(self.audio_features, n_delta=1, N=2)
                inp_res = tf.reshape(inp, [-1, self.audio_feat_dim * 2])
                layer_1 = tf.matmul(inp_res, weights_1) + biases_1
                layer_1a = tf.nn.leaky_relu(layer_1, alpha=0.3)
                layer_2 = tf.matmul(layer_1a, weights_2) + biases_2
                layer_2a = tf.nn.leaky_relu(layer_2, alpha=0.3)
                layer_3 = tf.matmul(layer_2a, weights_3) + biases_3
                
                out_res = tf.reshape(layer_3, [tf.shape(self.audio_features)[0], tf.shape(self.audio_features)[1], 200])
                # Apply mask
                #active_bins_frame_count = tf.reduce_sum(tf.cast(self.target_spec_norm > 0.1, dtype=tf.float32), axis=2)
                #active_frame_mask = tf.cast(active_bins_frame_count > 10, dtype=tf.float32)
                #emb_mask = self.masks[:, :, 0] * active_frame_mask
                emb_mask = self.masks[:, :, 0]
                self.speaker_embedding_ext = tf.multiply(out_res, tf.expand_dims(emb_mask, axis=2), name='speaker_embedding_ext')
                speaker_embedding_avg = tf.div(tf.reduce_sum(self.speaker_embedding_ext, axis=1),
                                               tf.expand_dims(tf.reduce_sum(emb_mask, axis=1) + 1, axis=1), name='speaker_embedding')
                #speaker_embedding_avg_squared = tf.reduce_sum(tf.square(self.speaker_embedding_ext), axis=1) / tf.expand_dims(tf.reduce_sum(emb_mask, axis=1), axis=1)
                #speaker_embedding_std = tf.sqrt(speaker_embedding_avg_squared - tf.square(speaker_embedding_avg))
                #self._speaker_embedding = tf.nn.sigmoid(tf.matmul(tf.concat([speaker_embedding_avg, speaker_embedding_std], axis=1), weights_4) + biases_4)
                #self._speaker_embedding = tf.concat([speaker_embedding_avg, speaker_embedding_std], axis=1)
                self._speaker_embedding = speaker_embedding_avg

        return self._speaker_embedding

    
    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            speaker_embedding_tiles = tf.tile(tf.expand_dims(self.speaker_embedding, axis=1), [1, max_sequence_length, 1])

            if self.int_layer == 0:
                inputs = tf.concat([self.net_inputs, speaker_embedding_tiles], axis=2)

                if self.is_training:
                    stacked_blstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                        num_layers=self.num_layers,
                                                        num_units=self.net_dim[0],
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.0,
                                                        seed=0)
                    t_net_inputs = tf.transpose(inputs, [1, 0, 2])
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
                             inputs=inputs,
                             dtype=tf.float32)
                rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)
            else:
                if self.is_training:
                    t_net_inputs = tf.transpose(self.net_inputs, [1, 0, 2])
                    t_spk_emb = tf.transpose(speaker_embedding_tiles, [1, 0, 2])
                    
                    with tf.variable_scope('blstm_1'):
                        blstm_1 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                            num_layers=self.int_layer,
                                                            num_units=self.net_dim[0],
                                                            input_mode='linear_input',
                                                            direction='bidirectional',
                                                            dropout=0.0,
                                                            seed=0)
                        t_rnn_outputs_1, _ = blstm_1(inputs=t_net_inputs)

                    t_inputs_spk_adapt = tf.concat([t_rnn_outputs_1, t_spk_emb], axis=2)
                    with tf.variable_scope('blstm_2'):
                        blstm_2 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                        num_layers=self.num_layers-self.int_layer,
                                                        num_units=self.net_dim[1],
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.0,
                                                        seed=0)
                        t_rnn_outputs_2, _ = blstm_2(inputs=t_inputs_spk_adapt)
                        rnn_outputs_2 = tf.transpose(t_rnn_outputs_2, [1, 0, 2])
                else:
                    blstm_cell = lambda x: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(x)
                    with tf.variable_scope('blstm_1/cudnn_lstm'):
                        forward_cells_1 = [blstm_cell(self.net_dim[0])]
                        backward_cells_1 = [blstm_cell(self.net_dim[0])]
    
                        rnn_outputs_1, output_state_fw_1, output_state_bw_1 = \
                            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=forward_cells_1,
                                cells_bw=backward_cells_1,
                                inputs=self.net_inputs,
                                dtype=tf.float32)

                    inputs_spk_adapt = tf.concat([rnn_outputs_1, speaker_embedding_tiles], axis=2)
                    with tf.variable_scope('blstm_2/cudnn_lstm'):
                        forward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                        backward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                        rnn_outputs_2, output_state_fw_2, output_state_bw_2 = \
                            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=forward_cells_2,
                                cells_bw=backward_cells_2,
                                inputs=self.net_inputs,
                                dtype=tf.float32)
                rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs_2, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)

            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

        return self._inference

    """
    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            speaker_embedding_tiles = tf.tile(tf.expand_dims(self.speaker_embedding, axis=1), [1, max_sequence_length, 1])
            
            if self.is_training:
                t_net_inputs = tf.transpose(self.net_inputs, [1, 0, 2])
                t_spk_emb = tf.transpose(speaker_embedding_tiles, [1, 0, 2])
                
                with tf.variable_scope('blstm_1'):
                    blstm_1 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                        num_layers=1,
                                                        num_units=self.net_dim[0],
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.0,
                                                        seed=0)
                    t_rnn_outputs_1, _ = blstm_1(inputs=t_net_inputs)

                t_inputs_spk_adapt = tf.concat([t_rnn_outputs_1, t_spk_emb], axis=2)
                with tf.variable_scope('blstm_2'):
                    blstm_2 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                    num_layers=self.num_layers-1,
                                                    num_units=self.net_dim[1],
                                                    input_mode='linear_input',
                                                    direction='bidirectional',
                                                    dropout=0.0,
                                                    seed=0)
                    t_rnn_outputs_2, _ = blstm_2(inputs=t_inputs_spk_adapt)
                    rnn_outputs_2 = tf.transpose(t_rnn_outputs_2, [1, 0, 2])
            else:
                blstm_cell = lambda x: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(x)
                with tf.variable_scope('blstm_1/cudnn_lstm'):
                    forward_cells_1 = [blstm_cell(self.net_dim[0])]
                    backward_cells_1 = [blstm_cell(self.net_dim[0])]
    
                    rnn_outputs_1, output_state_fw_1, output_state_bw_1 = \
                        tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw=forward_cells_1,
                            cells_bw=backward_cells_1,
                            inputs=self.net_inputs,
                            dtype=tf.float32)

                inputs_spk_adapt = tf.concat([rnn_outputs_1, speaker_embedding_tiles], axis=2)
                with tf.variable_scope('blstm_2/cudnn_lstm'):
                    forward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                    backward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                    rnn_outputs_2, output_state_fw_2, output_state_bw_2 = \
                        tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw=forward_cells_2,
                            cells_bw=backward_cells_2,
                            inputs=self.net_inputs,
                            dtype=tf.float32)


            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs_2, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)
            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

        return self._inference
    """

    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            # Restore original non-masked TF-bins
            prediction = self.target_spec_norm * self.masks + self.inference * (1 - self.masks) 
            #prediction = self.inference
            prediction = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            # L1 loss
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            # L2 loss
            #self.loss_hole = tf.identity(tf.nn.l2_loss((self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            #self.loss_valid = tf.identity(tf.nn.l2_loss((self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            #self.loss_func = tf.add(6 * self.loss_hole, self.loss_valid, name='func_loss')
            self.loss_func = self.loss_hole
            #self.loss_func = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                speaker_embeddings = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.speaker_embedding_ext, [0, 2, 1]), axis=3))
                tf.summary.image('Speaker_embeddings', speaker_embeddings, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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


class StackedBLSTMEmbeddingModel(object):
    """
    Speech inpainting BLSTM model
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: stacked BLSTM.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
    """

    def __init__(self, sequence_lengths, target_sources, masks, audio_feat_mean, audio_feat_std, dropout_rate, config,
                 audio_features=None, video_features=None, embeddings=None, input='a', is_training=True):
        self.sequence_lengths = sequence_lengths
        self.audio_feat_dim = config['audio_feat_dim']
        self.audio_len = config['audio_len']
        self.target_sources = target_sources
        self.video_features = video_features
        self.embeddings = embeddings
        self.masks = masks
        self.audio_feat_mean = audio_feat_mean
        self.audio_feat_std = audio_feat_std
        stft_shape = tf.convert_to_tensor([tf.shape(target_sources)[0], tf.reduce_max(sequence_lengths), self.audio_feat_dim])
        self.target_stft = get_stft(target_sources, window_size=24, step_size=12, n_fft=512, out_shape=stft_shape)
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        if audio_features is None:
            self.audio_features = self.target_spec_norm * masks
        else:
            self.audio_features = audio_features
        # input selection
        self.input_type = input
        if self.input_type == 'a':
            self.net_inputs = self.audio_features
        elif self.input_type == 'v':
            self.net_inputs = self.video_features
        elif self.input_type == 'av':
            self.net_inputs = tf.concat([self.audio_features, self.video_features], axis=2)
        self.dropout_rate = dropout_rate
        self.net_dim = config['net_dim']
        self.int_layer = config['integration_layer']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._summaries = None
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
        self.prediction
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
        self.var_scope = var_scope
        self.all_vars
        self.train_vars
    
    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            # replicate embedding for each frame
            embeddings_tiles = tf.tile(tf.expand_dims(self.embeddings, axis=1), [1, max_sequence_length, 1])
            inputs = tf.concat([self.net_inputs, embeddings_tiles], axis=2)

            if self.int_layer == 0:
                inputs = tf.concat([self.net_inputs, embeddings_tiles], axis=2)

                if self.is_training:
                    stacked_blstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                        num_layers=self.num_layers,
                                                        num_units=self.net_dim[0],
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.0,
                                                        seed=0)
                    t_net_inputs = tf.transpose(inputs, [1, 0, 2])
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
                             inputs=inputs,
                             dtype=tf.float32)
                rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)
            else:
                if self.is_training:
                    t_net_inputs = tf.transpose(self.net_inputs, [1, 0, 2])
                    t_spk_emb = tf.transpose(embeddings_tiles, [1, 0, 2])
                    
                    with tf.variable_scope('blstm_1'):
                        blstm_1 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                            num_layers=self.int_layer,
                                                            num_units=self.net_dim[0],
                                                            input_mode='linear_input',
                                                            direction='bidirectional',
                                                            dropout=0.0,
                                                            seed=0)
                        t_rnn_outputs_1, _ = blstm_1(inputs=t_net_inputs)

                    t_inputs_spk_adapt = tf.concat([t_rnn_outputs_1, t_spk_emb], axis=2)
                    with tf.variable_scope('blstm_2'):
                        blstm_2 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                        num_layers=self.num_layers-self.int_layer,
                                                        num_units=self.net_dim[1],
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.0,
                                                        seed=0)
                        t_rnn_outputs_2, _ = blstm_2(inputs=t_inputs_spk_adapt)
                        rnn_outputs_2 = tf.transpose(t_rnn_outputs_2, [1, 0, 2])
                else:
                    blstm_cell = lambda x: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(x)
                    with tf.variable_scope('blstm_1/cudnn_lstm'):
                        forward_cells_1 = [blstm_cell(self.net_dim[0])]
                        backward_cells_1 = [blstm_cell(self.net_dim[0])]
    
                        rnn_outputs_1, output_state_fw_1, output_state_bw_1 = \
                            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=forward_cells_1,
                                cells_bw=backward_cells_1,
                                inputs=self.net_inputs,
                                dtype=tf.float32)

                    inputs_spk_adapt = tf.concat([rnn_outputs_1, embeddings_tiles], axis=2)
                    with tf.variable_scope('blstm_2/cudnn_lstm'):
                        forward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                        backward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                        rnn_outputs_2, output_state_fw_2, output_state_bw_2 = \
                            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=forward_cells_2,
                                cells_bw=backward_cells_2,
                                inputs=self.net_inputs,
                                dtype=tf.float32)
                rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs_2, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)

            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

        return self._inference
    """

    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            embeddings_tiles = tf.tile(tf.expand_dims(self.embeddings, axis=1), [1, max_sequence_length, 1])
            
            if self.is_training:
                t_net_inputs = tf.transpose(self.net_inputs, [1, 0, 2])
                t_embeddings_tiles = tf.transpose(embeddings_tiles, [1, 0, 2])
                
                with tf.variable_scope('blstm_1'):
                    blstm_1 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                        num_layers=1,
                                                        num_units=self.net_dim[0],
                                                        input_mode='linear_input',
                                                        direction='bidirectional',
                                                        dropout=0.0,
                                                        seed=0)
                    t_rnn_outputs_1, _ = blstm_1(inputs=t_net_inputs)

                t_inputs_spk_adapt = tf.concat([t_rnn_outputs_1, t_embeddings_tiles], axis=2)
                with tf.variable_scope('blstm_2'):
                    blstm_2 = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                    num_layers=self.num_layers-1,
                                                    num_units=self.net_dim[1],
                                                    input_mode='linear_input',
                                                    direction='bidirectional',
                                                    dropout=0.0,
                                                    seed=0)
                    t_rnn_outputs_2, _ = blstm_2(inputs=t_inputs_spk_adapt)
                    rnn_outputs_2 = tf.transpose(t_rnn_outputs_2, [1, 0, 2])
            else:
                blstm_cell = lambda x: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(x)
                with tf.variable_scope('blstm_1/cudnn_lstm'):
                    forward_cells_1 = [blstm_cell(self.net_dim[0])]
                    backward_cells_1 = [blstm_cell(self.net_dim[0])]
    
                    rnn_outputs_1, output_state_fw_1, output_state_bw_1 = \
                        tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw=forward_cells_1,
                            cells_bw=backward_cells_1,
                            inputs=self.net_inputs,
                            dtype=tf.float32)

                inputs_spk_adapt = tf.concat([rnn_outputs_1, embeddings_tiles], axis=2)
                with tf.variable_scope('blstm_2/cudnn_lstm'):
                    forward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                    backward_cells_2 = [blstm_cell(self.net_dim[i]) for i in range(1, self.num_layers)]
                    rnn_outputs_2, output_state_fw_2, output_state_bw_2 = \
                        tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw=forward_cells_2,
                            cells_bw=backward_cells_2,
                            inputs=inputs_spk_adapt,
                            dtype=tf.float32)


            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs_2, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)
            with tf.variable_scope('logits'):
                weights = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits = tf.matmul(rnn_outputs_res, weights) + biases
            self._inference = tf.reshape(logits, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

        return self._inference
    """

    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            # Restore original non-masked TF-bins
            prediction = self.target_spec_norm * self.masks + self.inference * (1 - self.masks) 
            #prediction = self.inference
            prediction = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def loss(self):
        if self._loss is None:
            # L1 loss
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            # L2 loss
            #self.loss_hole = tf.identity(tf.nn.l2_loss((self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            #self.loss_valid = tf.identity(tf.nn.l2_loss((self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            #self.loss_func = tf.add(6 * self.loss_hole, self.loss_valid, name='func_loss')
            self.loss_func = self.loss_hole
            #self.loss_func = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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


class StackedBLSTMCTCLossModel(object):
    """
    Speech inpainting BLSTM model
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: stacked BLSTM.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
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
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        if audio_features is None:
            self.audio_features = self.target_spec_norm * masks
        else:
            self.audio_features = audio_features
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
        self.ctc_loss_weight = config['ctc_loss']
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._decoding = None
        self._per = None
        self._summaries = None
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
        self.prediction
        self.decoding
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
        self.var_scope = var_scope
        self.all_vars
        self.train_vars
        
    @property
    def inference(self):
        if self._inference is None:
            max_sequence_length = tf.reduce_max(self.sequence_lengths)
            speaker_embedding_tiles = tf.tile(tf.expand_dims(self.speaker_embedding, axis=1), [1, max_sequence_length, 1])
            inputs = tf.concat([self.net_inputs, speaker_embedding_tiles], axis=2)

            if self.is_training:
                stacked_blstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                                                    num_layers=self.num_layers,
                                                    num_units=self.net_dim[0],
                                                    input_mode='linear_input',
                                                    direction='bidirectional',
                                                    dropout=0.0,
                                                    seed=0)
                t_net_inputs = tf.transpose(inputs, [1, 0, 2])
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
                         inputs=inputs,
                         dtype=tf.float32)

            rnn_outputs_res = tf.nn.dropout(tf.reshape(rnn_outputs, [-1, self.net_dim[-1] * 2]), rate=self.dropout_rate)
            with tf.variable_scope('inpainting'):
                weights_ipt = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases_ipt = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits_ipt = tf.matmul(rnn_outputs_res, weights_ipt) + biases_ipt
                logits_ipt = tf.reshape(logits_ipt, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

            with tf.variable_scope('asr'):
                weights_asr = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.num_classes], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases_asr = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name='biases')
                logits_asr = tf.matmul(rnn_outputs_res, weights_asr) + biases_asr
                logits_asr = tf.reshape(logits_asr, [tf.shape(self.net_inputs)[0], -1, self.num_classes], name='inference')

            self._inference = logits_ipt, logits_asr

        return self._inference

    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            # Restore original non-masked TF-bins
            prediction = self.target_spec_norm * self.masks + self.inference[0] * (1 - self.masks) 
            #prediction = self.inference[0]
            prediction = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def decoding(self):
        if self._decoding is None:
            tm_logits = tf.transpose(self.inference[1], (1, 0, 2)) # CTC operations are defined as time major
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(tm_logits, self.sequence_lengths, beam_width=20)
            self.sparse_decoding = tf.cast(decoded[0], tf.int32, name='sparse_decoding')
            self._decoding = tf.sparse.to_dense(self.sparse_decoding, default_value=-1, name='decoding')
            
        return self._decoding

    @property
    def loss(self):
        if self._loss is None:
            # L1 loss
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            #self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            # CTC loss
            tm_logits = tf.transpose(self.inference[1], (1, 0, 2)) # CTC operations are defined as time major
            self.ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(self.sparse_labels, tm_logits, sequence_length=self.sequence_lengths,
                                                          preprocess_collapse_repeated=False, ctc_merge_repeated=True, 
                                                          time_major=True))
            #self.loss_hole = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            self.loss_func = self.loss_hole + self.ctc_loss_weight * self.ctc_loss
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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


class StackedBLSTMSSNNCTCLossModel(object):
    """
    Speech inpainting BLSTM model
    Input: log-compressed linear spectrogram of corrupted-audio.
    Model: stacked BLSTM.
    Output: log-compressed linear spectrogram of restored audio.
    Loss: L1 (target_spectrogram - reconstructed_spectrogram)
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
        self.target_spec = get_spectrogram(self.target_stft, log=True)
        self.target_spec_norm = (self.target_spec - audio_feat_mean) / audio_feat_std # standard normalization
        if audio_features is None:
            self.audio_features = self.target_spec_norm * masks
        else:
            self.audio_features = audio_features
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
        self.ctc_loss_weight = config['ctc_loss']
        self.net_dim = config['net_dim']
        # the last layer is always linear + softmax layer
        self.num_layers = len(self.net_dim)
        self.optimizer_choice = config['optimizer_type']
        self.starter_learning_rate = config['starter_learning_rate']
        self.updating_step = config['lr_updating_steps']
        self.learning_decay = config['lr_decay']
        self.is_training = is_training
        if is_training:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = tf.shape(target_sources)[0]
        self.regularization = config['l2']
        self._speaker_embedding = None
        self._inference = None
        self._loss = None
        self._train_op = None
        self._prediction = None
        self._enhanced_sources = None
        self._enhanced_sources_oracle_phase = None
        self._decoding = None
        self._per = None
        self._summaries = None
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
        self.speaker_embedding
        self.inference
        self.prediction
        self.decoding
        self.loss
        self.train_op
        self.enhanced_sources
        self.enhanced_sources_oracle_phase
        self.summaries
        self.var_scope = var_scope
        self.all_vars
        self.train_vars

    @property
    def speaker_embedding(self):
        if self._speaker_embedding is None:
            with tf.variable_scope('speaker_embedding'):
                weights_1 = tf.Variable(tf.truncated_normal([self.audio_feat_dim * 2, 200], stddev=1.0 / math.sqrt(self.audio_feat_dim),
                      dtype=tf.float32), name='weights_1')
                biases_1 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_1')
                weights_2 = tf.Variable(tf.truncated_normal([200, 200], stddev=1.0 / math.sqrt(200), dtype=tf.float32), name='weights_2')
                biases_2 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_2')
                weights_3 = tf.Variable(tf.truncated_normal([200, 200], stddev=1.0 / math.sqrt(200), dtype=tf.float32), name='weights_3')
                biases_3 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_3')
                #weights_4 = tf.Variable(tf.truncated_normal([400, 200], stddev=1.0 / math.sqrt(200), dtype=tf.float32), name='weights_4')
                #biases_4 = tf.Variable(tf.zeros([200], dtype=tf.float32), name='biases_4')

                #prev_spec_1 = tf.pad(self.audio_features, [[0, 0], [1,0], [0,0]], mode='CONSTANT')[:, :-1, :]
                #prev_spec_2 = tf.pad(self.audio_features, [[0, 0], [2,0], [0,0]], mode='CONSTANT')[:, :-2, :]
                #flw_spec_1 = tf.pad(self.audio_features, [[0, 0], [0,1], [0,0]], mode='CONSTANT')[:, 1:, :]
                #flw_spec_2 = tf.pad(self.audio_features, [[0, 0], [0,2], [0,0]], mode='CONSTANT')[:, 2:, :]
                #inp = tf.concat([prev_spec_2, prev_spec_1, self.audio_features, flw_spec_1, flw_spec_2], axis=2)
                inp = add_delta_features(self.audio_features, n_delta=1, N=2)
                inp_res = tf.reshape(inp, [-1, self.audio_feat_dim * 2])
                layer_1 = tf.matmul(inp_res, weights_1) + biases_1
                layer_1a = tf.nn.leaky_relu(layer_1, alpha=0.3)
                layer_2 = tf.matmul(layer_1a, weights_2) + biases_2
                layer_2a = tf.nn.leaky_relu(layer_2, alpha=0.3)
                layer_3 = tf.matmul(layer_2a, weights_3) + biases_3
                
                out_res = tf.reshape(layer_3, [tf.shape(self.audio_features)[0], tf.shape(self.audio_features)[1], 200])
                # Apply mask
                #active_bins_frame_count = tf.reduce_sum(tf.cast(self.target_spec_norm > 0.1, dtype=tf.float32), axis=2)
                #active_frame_mask = tf.cast(active_bins_frame_count > 10, dtype=tf.float32)
                #emb_mask = self.masks[:, :, 0] * active_frame_mask
                emb_mask = self.masks[:, :, 0]
                self.speaker_embedding_ext = tf.multiply(out_res, tf.expand_dims(emb_mask, axis=2), name='speaker_embedding_ext')
                speaker_embedding_avg = tf.div(tf.reduce_sum(self.speaker_embedding_ext, axis=1),
                                               tf.expand_dims(tf.reduce_sum(emb_mask, axis=1) + 1, axis=1), name='speaker_embedding')
                #speaker_embedding_avg_squared = tf.reduce_sum(tf.square(self.speaker_embedding_ext), axis=1) / tf.expand_dims(tf.reduce_sum(emb_mask, axis=1), axis=1)
                #speaker_embedding_std = tf.sqrt(speaker_embedding_avg_squared - tf.square(speaker_embedding_avg))
                #self._speaker_embedding = tf.nn.sigmoid(tf.matmul(tf.concat([speaker_embedding_avg, speaker_embedding_std], axis=1), weights_4) + biases_4)
                #self._speaker_embedding = tf.concat([speaker_embedding_avg, speaker_embedding_std], axis=1)
                self._speaker_embedding = speaker_embedding_avg

        return self._speaker_embedding

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
            with tf.variable_scope('inpainting'):
                weights_ipt = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.audio_feat_dim], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases_ipt = tf.Variable(tf.zeros([self.audio_feat_dim], dtype=tf.float32), name='biases')
                logits_ipt = tf.matmul(rnn_outputs_res, weights_ipt) + biases_ipt
                logits_ipt = tf.reshape(logits_ipt, [tf.shape(self.net_inputs)[0], max_sequence_length, self.audio_feat_dim], name='inference')

            with tf.variable_scope('asr'):
                weights_asr = tf.Variable(tf.truncated_normal([self.net_dim[-1] * 2, self.num_classes], stddev=1.0 / math.sqrt(float(self.net_dim[-1] * 2)),
                  dtype=tf.float32), name='weights')
                biases_asr = tf.Variable(tf.zeros([self.num_classes], dtype=tf.float32), name='biases')
                logits_asr = tf.matmul(rnn_outputs_res, weights_asr) + biases_asr
                logits_asr = tf.reshape(logits_asr, [tf.shape(self.net_inputs)[0], -1, self.num_classes], name='inference')

            self._inference = logits_ipt, logits_asr

        return self._inference

    @property
    def prediction(self):
        if self._prediction is None:
            # Inpaint corrupted TF bins only.
            #prediction = self.target_spec_norm + self.inference * (1 - self.mask)
            #self._prediction = tf.identity(prediction, name='prediction')
            # Restore original non-masked TF-bins
            prediction = self.target_spec_norm * self.masks + self.inference[0] * (1 - self.masks) 
            #prediction = self.inference[0]
            prediction = tf.expand_dims(tf.sequence_mask(self.sequence_lengths, dtype=tf.float32), axis=2) * prediction
            self._prediction = tf.identity(prediction, name='prediction')
        return self._prediction

    @property
    def decoding(self):
        if self._decoding is None:
            tm_logits = tf.transpose(self.inference[1], (1, 0, 2)) # CTC operations are defined as time major
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(tm_logits, self.sequence_lengths, beam_width=20)
            self.sparse_decoding = tf.cast(decoded[0], tf.int32, name='sparse_decoding')
            self._decoding = tf.sparse.to_dense(self.sparse_decoding, default_value=-1, name='decoding')
            
        return self._decoding

    @property
    def loss(self):
        if self._loss is None:
            # L1 loss
            self.loss_hole = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * (1 - self.masks)) / tf.reduce_sum(1 - self.masks), name='loss_hole')
            #self.loss_valid = tf.identity(tf.reduce_sum(tf.abs(self.target_spec_norm - self.prediction) * self.masks) / tf.reduce_sum(self.masks), name='loss_valid')
            # CTC loss
            tm_logits = tf.transpose(self.inference[1], (1, 0, 2)) # CTC operations are defined as time major
            self.ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(self.sparse_labels, tm_logits, sequence_length=self.sequence_lengths,
                                                          preprocess_collapse_repeated=False, ctc_merge_repeated=True, 
                                                          time_major=True))
            #self.loss_hole = tf.reduce_mean(tf.abs(self.target_spec_norm - self.prediction))
            self.loss_func = self.loss_hole + self.ctc_loss_weight * self.ctc_loss
            # Add regularization
            if self.regularization:
                self.reg_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.train_vars], name='reg_loss')
            else:
                self.reg_loss = tf.constant(0, dtype=tf.float32)
            # Compute final loss function
            self._loss = tf.add(self.loss_func, self.regularization * self.reg_loss, name='loss')
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
    def enhanced_sources(self):
        if self._enhanced_sources is None:
            # denormalize
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            masked_phase = tf.angle(self.target_stft * tf.cast(self.masks, dtype=tf.complex64))
            self._enhanced_sources = tf.identity(get_sources(mag_specs, masked_phase, num_samples=self.audio_len),
                                                             name='enhanced_sources')
        return self._enhanced_sources

    @property
    def enhanced_sources_oracle_phase(self):
        if self._enhanced_sources_oracle_phase is None:
            mag_specs = tf.exp(self.prediction * self.audio_feat_std + self.audio_feat_mean)
            self._enhanced_sources_oracle_phase = tf.identity(get_sources(mag_specs, tf.angle(self.target_stft), num_samples=self.audio_len),
                                                                          name='enhanced_sources_oracle_phase')
        return self._enhanced_sources_oracle_phase


    @property
    def summaries(self):
        """
        Add summaries for TensorBoard
        """
        if self._summaries is None:
            n_samples = 10
            with tf.name_scope('summary'):
                mag_target_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.target_spec_norm, [0, 2, 1]), axis=3))
                mag_enhanced_specs = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.prediction, [0, 2, 1]), axis=3))
                masks = tf.map_fn(tf.image.flip_up_down, tf.expand_dims(tf.transpose(self.masks, [0, 2, 1]), axis=3))
                tf.summary.image('Target_spectrogram', mag_target_specs, max_outputs=n_samples)
                tf.summary.image('Enhanced_spectrogram', mag_enhanced_specs, max_outputs=n_samples)
                tf.summary.image('Mask', masks, max_outputs=n_samples)
                target_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.target_sources), axis=1), (-1, 1))
                enhanced_sources_max = tf.reshape(tf.reduce_max(tf.abs(self.enhanced_sources), axis=1), (-1, 1))
                tf.summary.audio('Target_audio', self.target_sources / target_sources_max, 16000.0, max_outputs=n_samples)
                tf.summary.audio('Enhanced_audio', self.enhanced_sources / enhanced_sources_max, 16000.0, max_outputs=n_samples)
                self._summaries = tf.summary.merge_all()
        return self._summaries

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