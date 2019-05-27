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


class RNNLayers:
    def __init__(self, is_trainning, **kwargs):
        time_slices = kwargs.get('time_slices')
        self.x = tf.placeholder(tf.float32, [kwargs.get('mfcc_features'), time_slices])
        self.y = tf.placeholder(tf.float32, [kwargs.get('classes')])

        self.end = False

        x = tf.transpose(self.x, [1, 0])

        # RNN层
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=RNN_HIDDENSIZE, state_tuple=True)
        rnn_layers = [tf.contrib.rnn.DropoutWrapper(
            rnn_cell,
            output_keep_prob=1 - RNN_HIDDENSIZE
        )] * RNN_LAYERS if is_trainning else [rnn_cell] * RNN_LAYERS

        rnn_block = tf.contrib.rnn.MultiRNNCell(rnn_layers)

        self.rnn_init_state = rnn_block.zero_state(BATCH_SIZE, tf.float32)

        rnn_outputs = []
        rnn_state = self.rnn_init_state
        with tf.variable_scope('RNN'):
            for step in range(time_slices):
                if step > 1:
                    tf.get_variable_scope().reuse_variables()
                output, rnn_state = rnn_block(x[step, :], rnn_state)
                rnn_outputs.append(output)

        rnn_output = tf.reshape(tf.concat(rnn_outputs, axis=0), [])
