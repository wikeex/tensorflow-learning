import tensorflow as tf

HIDDEN_SIZE = 60
NUM_LAYERS = 4
LEARNING_RATE = 0.001
KEEP_PROB = 0.5


class SoundWatching:

    def __init__(self, is_training, batch_size, num_steps):
        self.x = tf.placeholder(tf.float32, [-1, 512, 80])

        x = tf.transpose(self.x, [0, 2, 1])

        layer_a = []

        for _ in range(NUM_LAYERS):
            lstm_cell_a = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True)

            if is_training:
                lstm_cell_a = tf.contrib.rnn.DropoutWrapper(lstm_cell_a, output_keep_prob=KEEP_PROB)

                layer_a.append(lstm_cell_a)

        rnn_net_a = tf.contrib.rnn.MultiRNNCell(layer_a)

        self.initial_state = rnn_net_a.zero_state(batch_size, tf.float32)

        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_output, state = rnn_net_a(x[:, step, :], state)
                outputs.append(rnn_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE*num_steps])

        softmax_weight = tf.get_variable('softmax_weight', [HIDDEN_SIZE*num_steps, 59])
        softmax_bias = tf.get_variable('softmax_bias', [59])

        logits = tf.matmul(output, softmax_weight) + softmax_bias
        logits_softmax = tf.nn.softmax(logits=logits, axis=1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y))

        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        self.correct_prediction = tf.equal(tf.math.argmax(self.y, 1), tf.math.argmax(logits_softmax, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if not is_training:
            return

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)