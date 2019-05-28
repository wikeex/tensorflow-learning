import tensorflow as tf


RNN_HIDDENSIZE = 512
LSTM1_HIDDENSIZE = 128
LSTM2_HIDDENSIZE = 64

BATCH_SIZE = 1

RNN_LAYERS = 2
LSTM1_LAYERS = 2
LSTM2_LAYERS = 2

RNN_RATE = 0.5
LSTM1_RATE = 0.5
LSTM2_RATE = 0.5

LEARNING_STEP = 0.001


class SoundLayers:
    def __init__(self, is_trainning, **kwargs):
        time_slices = kwargs.get('time_slices')
        features = kwargs.get('mfcc_features')
        classes = kwargs.get('classes')

        lstm1_block_output = []
        for lstm1_time_step in range(3):
            rnn_block_output = []
            with tf.variable_scope('rnn_block'):
                # rnn层处理0.5s的信息，6个rnn层的输出拼接成3s信息送往LSTM1层
                for rnn_time_step in range(6):
                    end = tf.placeholder(tf.bool, [1])
                    if end is True:
                        rnn_block_output.append(tf.Variable(tf.zeros([RNN_HIDDENSIZE])))
                        continue

                    self.x = tf.placeholder(tf.float32, [features, time_slices])
                    self.y = tf.placeholder(tf.float32, [classes])

                    if rnn_time_step > 1:
                        tf.get_variable_scope().reuse_variables()

                    x = tf.transpose(self.x, [1, 0])

                    # RNN层
                    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=RNN_HIDDENSIZE, state_tuple=True)
                    rnn_layers = [tf.contrib.rnn.DropoutWrapper(
                        rnn_cell,
                        output_keep_prob=1 - RNN_RATE
                    )] * RNN_LAYERS if is_trainning else [rnn_cell] * RNN_LAYERS

                    rnn_block = tf.contrib.rnn.MultiRNNCell(rnn_layers)

                    self.rnn_init_state = rnn_block.zero_state(BATCH_SIZE, tf.float32)

                    rnn_layers_outputs = []
                    rnn_state = self.rnn_init_state
                    with tf.variable_scope('rnn_layer'):
                        for step in range(time_slices):
                            if step > 1:
                                tf.get_variable_scope().reuse_variables()
                            output, rnn_state = rnn_block(x[step, :], rnn_state)
                            rnn_layers_outputs.append(output)

                    rnn_block_output.extend(rnn_layers_outputs)

            rnn_output = tf.reshape(tf.concat(rnn_block_output, axis=0), [-1, RNN_HIDDENSIZE])

            with tf.variable_scope('lstm1_block'):
                # lstm1层处理3s的信息，3个lstm1层输出拼接成9s信息送往lstm2层
                lstm1_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM1_HIDDENSIZE, state_is_tuple=True)
                lstm1_layers = [tf.contrib.rnn.DropoutWrapper(
                    lstm1_cell,
                    output_keep_prob=1 - LSTM1_RATE
                )] * LSTM1_LAYERS if is_trainning else [lstm1_cell] * LSTM1_LAYERS

                lstm1_block = tf.contrib.rnn.MultiRNNCell(lstm1_layers)

                self.lstm_init_state = lstm1_block.zero_state(BATCH_SIZE, tf.float32)

                lstm1_layers_outputs = []
                lstm1_state = self.lstm_init_state
                with tf.variable_scope('lstm1'):
                    for step in range(rnn_output.shape[0].value):
                        if step > 1:
                            tf.get_variable_scope().reuse_variables()
                        output, lstm1_state = lstm1_block(rnn_output[step, :], lstm1_state)
                        lstm1_layers_outputs.append(output)
                lstm1_block_output.extend(lstm1_layers_outputs)

                if end is True:
                    lstm1_block_output.extend(tf.Variable(tf.zeros([LSTM1_HIDDENSIZE])))