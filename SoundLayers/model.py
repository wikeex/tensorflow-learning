import tensorflow as tf


RNN_HIDDENSIZE = 512
LSTM1_HIDDENSIZE = 128
LSTM2_HIDDENSIZE = 64

BATCH_SIZE = 1

RNN_LAYERS = 2
LSTM1_LAYERS = 1
LSTM2_LAYERS = 1

RNN_RATE = 0.5
LSTM1_RATE = 0.5
LSTM2_RATE = 0.5

LEARNING_STEP = 0.001


class SoundLayers:
    def __init__(self, is_training, **kwargs):
        time_slices = kwargs.get('time_slices')
        features = kwargs.get('mfcc_features')
        classes = kwargs.get('classes')

        # rnn层处理0.5s的信息，6个rnn层的输出拼接成3s信息送往LSTM1层
        self.end = tf.placeholder(tf.bool, [1])
        end = self.end

        self.x = tf.placeholder(tf.float32, [features, time_slices])
        self.y = tf.placeholder(tf.float32, [classes])
        self.rnn_block_output = tf.placeholder(tf.float32)
        self.lstm1_block_output = tf.placeholder(tf.float32)

        x = tf.transpose(self.x, [1, 0])

        # RNN层
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=RNN_HIDDENSIZE)
        rnn_layers = [tf.contrib.rnn.DropoutWrapper(
            rnn_cell,
            output_keep_prob=1 - RNN_RATE
        )] * RNN_LAYERS if is_training else [rnn_cell] * RNN_LAYERS

        rnn_block = tf.contrib.rnn.MultiRNNCell(rnn_layers)

        self.rnn_init_state = rnn_block.zero_state(BATCH_SIZE, tf.float32)

        rnn_layers_outputs = []
        rnn_state = self.rnn_init_state

        with tf.variable_scope('rnn_layer'):
            for step in range(time_slices):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                output, rnn_state = rnn_block(tf.expand_dims(x[step, :], axis=0), rnn_state)
                rnn_layers_outputs.append(output)
        rnn_output = tf.concat(rnn_layers_outputs, axis=0)

        self.rnn_block_output = tf.concat([self.rnn_block_output, rnn_output])

        if end is True:
            rnn_rest = 60 - self.rnn_block_output.shape[0].value
            self.rnn_block_output = tf.pad(self.rnn_block_output, [0, rnn_rest], mode='CONSTANT')

        if self.rnn_block_output.shape[0].value >= 60:

            # lstm1层处理3s的信息，3个lstm1层输出拼接成9s信息送往lstm2层
            lstm1_cell = tf.contrib.rnn.BasicLSTMCell(num_units=LSTM1_HIDDENSIZE, state_is_tuple=True)
            lstm1_layers = [tf.contrib.rnn.DropoutWrapper(
                lstm1_cell,
                output_keep_prob=1 - LSTM1_RATE
            )] * LSTM1_LAYERS if is_training else [lstm1_cell] * LSTM1_LAYERS

            lstm1_block = tf.contrib.rnn.MultiRNNCell(lstm1_layers)

            self.lstm1_init_state = lstm1_block.zero_state(BATCH_SIZE, tf.float32)

            lstm1_layers_outputs = []
            lstm1_state = self.lstm1_init_state

            with tf.variable_scope('lstm1_layer'):
                for step in range(self.rnn_block_output.shape[0].value):
                    if step > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, lstm1_state = lstm1_block(tf.expand_dims(self.rnn_block_output[step, :], 0), lstm1_state)
                    lstm1_layers_outputs.append(output)
            lstm1_output = tf.concat(rnn_layers_outputs, axis=0)
            self.lstm1_block_output = tf.concat([self.lstm1_block_output, lstm1_output])
            if self.end is True:
                lstm1_rest = 180 - self.lstm1_block_output.shape[0].value
                self.lstm1_block_output = tf.pad(self.rnn_block_output, [0, lstm1_rest], mode='CONSTANT')

            if self.lstm1_block_output.shape[0].value >= 180:

                lstm1_output = tf.reshape(tf.concat(self.lstm1_block_output, axis=0), [-1, LSTM1_HIDDENSIZE])

                lstm2_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM2_HIDDENSIZE, state_is_tuple=True)

                # 可能需要每一层单独建立
                lstm2_layers = [tf.contrib.rnn.DropoutWrapper(
                    lstm2_cell,
                    output_keep_prob=1 - LSTM2_RATE
                )] * LSTM2_LAYERS if is_training else [lstm2_cell] * LSTM2_LAYERS

                lstm2_block = tf.contrib.rnn.MultiRNNCell(lstm2_layers)

                self.lstm2_init_state = lstm2_block.zero_state(BATCH_SIZE, tf.float32)

                lstm2_layers_outputs = []
                lstm2_state = self.lstm2_init_state
                lstm2_steps = lstm1_output.shape[0].value
                with tf.variable_scope('lstm2_layer'):
                    for step in range(lstm2_steps):
                        if step > 1:
                            tf.get_variable_scope().reuse_variables()
                        output, lstm2_state = lstm2_block(tf.expand_dims(lstm1_output[step, :], 0), lstm2_state)
                        lstm2_layers_outputs.append(output)

                lstm2_output = tf.reshape(
                    tf.concat(lstm2_layers_outputs, axis=0),
                    [-1, LSTM2_HIDDENSIZE * lstm2_steps]
                )

                softmax_weight = tf.get_variable('softmax_weight', [LSTM2_HIDDENSIZE * lstm2_steps, classes])
                softmax_bias = tf.get_variable('softmax_bias', [classes])

                logits = tf.matmul(lstm2_output, softmax_weight) + softmax_bias
                logits_softmax = tf.nn.softmax(logits)

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.y, logits))

                self.cost = tf.reduce_sum(loss)

                self.correct_prediction = tf.equal(tf.math.argmax(self.y, 1), tf.math.argmax(logits_softmax, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

                if not is_training:
                    return

                self.optimizer = tf.train.AdamOptimizer(LEARNING_STEP).minimize(loss)
