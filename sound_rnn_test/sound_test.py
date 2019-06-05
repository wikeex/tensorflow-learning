import tensorflow as tf
from sound_lstm_test import data
from sound_lstm_test.model import SoundTestModel

DATA_PATH = './datasets/numbers'

TRAIN_BATCH_SIZE = 1
TRAIN_NUM_STEPS = 10

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEPS = 10
NUM_EPOCH = 20


def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_accuracy = 0.0
    state = session.run(model.initial_state)

    writer = tf.summary.FileWriter('.')

    for step in range(epoch_size):
        x, y, end = next(data)
        cost, state, _, accuracy, merged = session.run(
            [model.cost, model.final_state, train_op, model.accuracy, model.merged],
            {model.x: x, model.y: y, model.initial_state: state}
        )
        total_accuracy += accuracy

        if output_log and step % 100 == 0:
            writer.add_summary(merged)
            with open('./recode.txt', 'a') as f:
                f.write('After %d steps, accuracy is %.3f\n' % (step, accuracy))
            print('After %d steps, accuracy is %.3f\n' % (step,  accuracy))
    print(total_accuracy, epoch_size)
    return total_accuracy / epoch_size


def main():
    train_data = data.np_load(batch_type='train/')
    valid_data = data.np_load(batch_type='eval/')
    test_data = data.np_load(batch_type='test/')

    train_epoch_size = 6000

    valid_epoch_size = 500

    test_epoch_size = 2000

    restore_check_point = True
    check_point_path = './model/sound_test'

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('sound_test_model', reuse=None, initializer=initializer):
        train_model = SoundTestModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEPS)

    with tf.variable_scope('sound_test_model', reuse=True, initializer=initializer):
        eval_model = SoundTestModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEPS)

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
            run_epoch(session, train_model, train_data, train_model.optimizer, True, train_epoch_size)

            valid_accuracy = run_epoch(session, eval_model, valid_data, tf.no_op(), False, valid_epoch_size)
            with open('./record.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print('Epoch: %d Validation Accuracy: %.3f' % (i + 1, valid_accuracy))

            if valid_accuracy > best_accuracy:
                saver.save(session, check_point_path)

        test_accuracy = run_epoch(session, eval_model, test_data, tf.no_op(), False, test_epoch_size)
        with open('./record.txt', 'a') as f:
            f.write('In iteration: %d\n' % (i + 1))
        print('Test Accuracy: %.3f' % test_accuracy)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()



