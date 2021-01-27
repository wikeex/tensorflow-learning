import tensorflow as tf

HIDDEN_SIZE = 80
NUM_LAYERS = 4
LEARNING_RATE = 0.0001
KEEP_PROB = 0.8


class SoundTestModel:
    def __init__(self, is_training, batch_size, num_steps):
        self.x = tf.placeholder(tf.float32, [512, 10])
        self.y = tf.placeholder(tf.float32, [59])

        x = tf.transpose(self.x, [1, 0])
        y = tf.expand_dims(self.y, axis=0)

        cells = []

        for _ in range(NUM_LAYERS):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=HIDDEN_SIZE)

            if is_training:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=KEEP_PROB)

            cells.append(rnn_cell)

        rnn_net = tf.contrib.rnn.MultiRNNCell(cells)

        self.initial_state = rnn_net.zero_state(batch_size, tf.float32)

        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for step in range(num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                rnn_output, state = rnn_net(tf.expand_dims(x[step, :], axis=0), state)
                outputs.append(rnn_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE*num_steps])

        softmax_weight = tf.get_variable('softmax_weight', [HIDDEN_SIZE*num_steps, 59])
        softmax_bias = tf.get_variable('softmax_bias', [59])

        logits = tf.matmul(output, softmax_weight) + softmax_bias
        logits_softmax = tf.nn.softmax(logits=logits, axis=1)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

        self.cost = tf.reduce_sum(loss) / batch_size
        cost_summary = tf.summary.scalar('cost', self.cost)

        self.final_state = state

        self.correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(logits_softmax, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge([cost_summary, accuracy_summary])

        if not is_training:
            return

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
