import tensorflow as tf
from sound_lstm_test import data

DATA_PATH = 'G:/sound_fixed'
FEATURES_SIZE = 512

HIDDEN_SIZE = 800
NUM_LAYERS = 3
LEARNING_RATE = 1.0
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

TRAIN_BATCH_SIZE = 10
TRAIN_NUM_STEP = 80

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 100


class SoundModel:
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.float32, [batch_size, 512, 80])
        self.targets = tf.placeholder(tf.float32, [batch_size, 59])

        inputs = tf.transpose(self.input_data, [0, 2, 1])

        cells = []
        for _ in range(NUM_LAYERS):
            # 基本lstm单元，隐含状态数和输出特征维度都为HIDDEN_SIZE
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True)
            if is_training:
                # 每个lstm单元外包裹一个DropoutWrapper
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)

            cells.append(lstm_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)  # 构建多层rnn网络结构

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        softmax_weight = tf.get_variable('softmax_w', [HIDDEN_SIZE, 59])
        softmax_bias = tf.get_variable('softmax_b', [59])

        logits = tf.matmul(cell_output, softmax_weight) + softmax_bias
        # 平方差损失
        # loss = tf.square(tf.subtract(self.targets, logits))

        # 交叉熵损失
        loss = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(logits), reduction_indices=1))

        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        self.correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if not is_training:
            return

        trainable_variable = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_variable), MAX_GRAD_NORM)
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variable))


def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_accuracy = 0.0
    state = session.run(model.initial_state)

    for step in range(epoch_size):
        x, y = next(data)
        cost, state, _, accuracy = session.run(
            [model.cost, model.final_state, train_op, model.accuracy],
            {model.input_data: x, model.targets: y, model.initial_state: state}
        )
        total_accuracy += accuracy

        if output_log and step % 100 == 0:
            with open('./recode.txt', 'a') as f:
                f.write('After %d steps, accuracy is %.3f\n' % (step, accuracy))
            print('After %d steps, accuracy is %.3f\n' % (step,  accuracy))
    print(total_accuracy, epoch_size)
    return total_accuracy / epoch_size


def main():
    train_data = data.np_load(10, 'train/')
    valid_data = data.np_load(1, 'eval/')
    test_data = data.np_load(1, 'test/')

    train_epoch_size = 6000

    valid_epoch_size = 500

    test_epoch_size = 2000

    restore_check_point = True
    check_point_path = './model/sound'

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('sound_model', reuse=None, initializer=initializer):
        train_model = SoundModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope('sound_model', reuse=True, initializer=initializer):
        eval_model = SoundModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    saver = tf.train.Saver()

    with tf.Session() as session:
        if restore_check_point and tf.train.checkpoint_exists(check_point_path):
            saver.restore(session, check_point_path)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        best_accuracy = 0
        for i in range(NUM_EPOCH):
            with open('./recode.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print('In iteration: %d' % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True, train_epoch_size)

            valid_accuracy = run_epoch(session, eval_model, valid_data, tf.no_op(), False, valid_epoch_size)
            with open('record.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print('Epoch: %d Validation Accuracy: %.3f' % (i + 1, valid_accuracy))

            if valid_accuracy > best_accuracy:
                saver.save(session, check_point_path)

        test_accuracy = run_epoch(session, eval_model, test_data, tf.no_op(), False, test_epoch_size)
        with open('record.txt', 'a') as f:
            f.write('In iteration: %d\n' % (i + 1))
        print('Test Accuracy: %.3f' % test_accuracy)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()

