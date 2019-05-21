import tensorflow as tf

FEATURES_SIZE = 20

HIDDEN_SIZE = 128
NUM_LAYERS = 4
LEARNING_RATE = 0.001
KEEP_PROB = 0.5

TRAIN_BATCH_SIZE = 10
TRAIN_NUM_STEPS = 80

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEPS = 1
NUM_EPOCH = 100


class SoundTestModel:
    def __init__(self, is_training, batch_size, num_steps):
        self.x = tf.placeholder(tf.float32, [batch_size, 20, 80])
        self.y = tf.placeholder(tf.float32, [batch_size, 10])

        x = tf.transpose(self.x, [0, 2, 1])

        cells = []

        for _ in range(NUM_LAYERS):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True)

            if is_training:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)

            cells.append(lstm_cell)

        rnn_net = tf.contrib.rnn.MultiRNNCell(cells)

        self.initial_state = rnn_net.zero_state(batch_size, tf.float32)

        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_output, state = rnn_net(x[:, step, :], state)
                outputs.append(rnn_output)

        softmax_weight = tf.get_variable('softmax_weight', [HIDDEN_SIZE, 10])
        softmax_bias = tf.get_variable('softmax_bias', [10])

        logits = tf.matmul(rnn_output, softmax_weight) + softmax_bias

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        self.correct_prediction = tf.equal(tf.arg_max(self.y, 1), tf.arg_max(logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if not is_training:
            return

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
