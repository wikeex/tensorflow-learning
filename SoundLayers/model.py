import tensorflow as tf


RNN_HIDDENSIZE = 100
LSTM1_HIDDENSIZE = 150
LSTM2_HIDDENSIZE = 200

BATCH_SIZE = 1

RNN_LAYERS = 4
LSTM1_LAYERS = 4
LSTM2_LAYERS = 4

RNN_RATE = 0.3
LSTM1_RATE = 0.4
LSTM2_RATE = 0.3

RNN_LEARNING_STEP = 0.0001
LSTM1_LEARNING_STEP = 0.0001
LSTM2_LEARNING_STEP = 0.0001


class RNNLayer:
    def __init__(self, is_training, **kwargs):
        time_slices = kwargs.get('time_slices')
        features = kwargs.get('mfcc_features')
        classes = kwargs.get('classes')

        # rnn层处理0.5s的信息，6个rnn层的输出拼接成3s信息送往LSTM1层

        self.x = tf.placeholder(tf.float32, [features, time_slices])
        self.y = tf.placeholder(tf.float32, [classes])

        x = tf.expand_dims(tf.transpose(self.x, [1, 0]), axis=0)
        y = tf.expand_dims(self.y, axis=0)

        # RNN层
        fw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=RNN_HIDDENSIZE)
        bw_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=RNN_HIDDENSIZE)
        if is_training:
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=1 - RNN_RATE)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=1 - RNN_RATE)

        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

        self.output = tf.concat(outputs, axis=0)
        logits_input = tf.reshape(self.output, [-1, 2 * RNN_HIDDENSIZE * time_slices])

        softmax_weight = tf.get_variable('rnn_softmax_weight', [2 * RNN_HIDDENSIZE * time_slices, classes])
        softmax_bias = tf.get_variable('rnn_softmax_bias', [classes])

        logits = tf.matmul(logits_input, softmax_weight) + softmax_bias
        logits_softmax = tf.nn.softmax(logits, axis=1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits))

        self.cost = tf.reduce_sum(loss)
        self.cost_summary = tf.summary.scalar('rnn_cost', self.cost)

        self.correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(logits_softmax, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy_summary = tf.summary.scalar('rnn_accuracy', self.accuracy)

        self.merged = tf.summary.merge([self.cost_summary, self.accuracy_summary])

        if not is_training:
            return

        self.optimizer = tf.train.AdamOptimizer(RNN_LEARNING_STEP).minimize(loss)


class LSTM1Layer:
    def __init__(self, is_training, **kwargs):
        features = kwargs.get('mfcc_features')
        time_slices = kwargs.get('time_slices')
        classes = kwargs.get('classes')

        x = self.x = tf.placeholder(dtype=tf.float32, shape=[time_slices, features])
        self.y = tf.placeholder(dtype=tf.float32, shape=[classes])

        y = tf.expand_dims(self.y, axis=0)

        # lstm1层处理3s的信息，3个lstm1层输出拼接成9s信息送往lstm2层
        lstm1_layers = []
        for _ in range(RNN_LAYERS):
            lstm1_cell = tf.contrib.rnn.BasicLSTMCell(num_units=LSTM1_HIDDENSIZE, state_is_tuple=True)
            if is_training:
                lstm1_cell = tf.contrib.rnn.DropoutWrapper(lstm1_cell, output_keep_prob=1 - RNN_RATE)
            lstm1_layers.append(lstm1_cell)

        lstm1_block = tf.contrib.rnn.MultiRNNCell(lstm1_layers)

        self.init_state = lstm1_block.zero_state(BATCH_SIZE, tf.float32)

        lstm1_layers_outputs = []
        lstm1_state = self.init_state

        with tf.variable_scope('lstm1_layer'):
            for step in range(time_slices):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                output, lstm1_state = lstm1_block(tf.expand_dims(x[step, :], 0), lstm1_state)
                lstm1_layers_outputs.append(output)
        self.output = tf.concat(lstm1_layers_outputs, axis=0)
        self.final_state = lstm1_state

        logits_input = tf.reshape(self.output, [-1, LSTM1_HIDDENSIZE * time_slices])

        softmax_weight = tf.get_variable('lstm1_softmax_weight', [LSTM1_HIDDENSIZE * time_slices, classes])
        softmax_bias = tf.get_variable('lstm1_softmax_bias', [classes])

        logits = tf.matmul(logits_input, softmax_weight) + softmax_bias
        logits_softmax = tf.nn.softmax(logits, axis=1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits))

        self.cost = tf.reduce_sum(loss)
        self.cost_summary = tf.summary.scalar('lstm1_cost', self.cost)

        self.correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(logits_softmax, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy_summary = tf.summary.scalar('lstm1_accuracy', self.accuracy)

        self.merged = tf.summary.merge([self.cost_summary, self.accuracy_summary])

        if not is_training:
            return

        self.optimizer = tf.train.AdamOptimizer(LSTM1_LEARNING_STEP).minimize(loss)


class LSTM2Layer:
    def __init__(self, is_training, **kwargs):
        features = kwargs.get('mfcc_features')
        time_slices = kwargs.get('time_slices')
        classes = kwargs.get('classes')

        x = self.x = tf.placeholder(dtype=tf.float32, shape=[time_slices, features])
        self.y = tf.placeholder(dtype=tf.float32, shape=[classes])

        y = tf.expand_dims(self.y, axis=0)

        # 可能需要每一层单独建立
        lstm2_layers = []
        for _ in range(RNN_LAYERS):
            lstm2_cell = tf.contrib.rnn.BasicLSTMCell(num_units=LSTM2_HIDDENSIZE, state_is_tuple=True)
            if is_training:
                lstm2_cell = tf.contrib.rnn.DropoutWrapper(lstm2_cell, output_keep_prob=1 - RNN_RATE)
            lstm2_layers.append(lstm2_cell)

        lstm2_block = tf.contrib.rnn.MultiRNNCell(lstm2_layers)

        self.init_state = lstm2_block.zero_state(BATCH_SIZE, tf.float32)

        lstm2_layers_outputs = []
        lstm2_state = self.init_state

        with tf.variable_scope('lstm2_layer'):
            for step in range(time_slices):
                if step > 1:
                    tf.get_variable_scope().reuse_variables()
                output, lstm2_state = lstm2_block(tf.expand_dims(x[step, :], 0), lstm2_state)
                lstm2_layers_outputs.append(output)
        self.final_state = lstm2_state

        self.output = tf.concat(lstm2_layers_outputs, axis=0)

        logits_input = tf.reshape(self.output, [-1, LSTM2_HIDDENSIZE * time_slices])

        softmax_weight = tf.get_variable('lstm2_softmax_weight', [LSTM2_HIDDENSIZE * time_slices, classes])
        softmax_bias = tf.get_variable('lstm2_softmax_bias', [classes])

        logits = tf.matmul(logits_input, softmax_weight) + softmax_bias
        logits_softmax = tf.nn.softmax(logits, axis=1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits))

        self.cost = tf.reduce_sum(loss)
        self.cost_summary = tf.summary.scalar('lstm2_cost', self.cost)

        self.correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(logits_softmax, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy_summary = tf.summary.scalar('lstm2_accuracy', self.accuracy)

        self.merged = tf.summary.merge([self.cost_summary, self.accuracy_summary])

        if not is_training:
            return

        self.optimizer = tf.train.AdamOptimizer(LSTM2_LEARNING_STEP).minimize(loss)


def BiRNN(x):
    lstm_fw_cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDENSIZE)
    lstm_bw_cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDENSIZE)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)