import tensorflow as tf
import numpy as np
from SoundLayers import data
from SoundLayers.model import RNNLayer, LSTM1Layer, LSTM2Layer


NUM_EPOCH = 500
RNN_ITERS = 0
LSTM1_ITERS = 0


def run_epoch(session, data, is_tranning, output_log, epoch_size, **kwargs):

    rnn_model = kwargs.get('rnn_model')
    lstm1_model = kwargs.get('lstm1_model')
    lstm2_model = kwargs.get('lstm2_model')

    rnn_total_accu = []
    lstm1_total_accu = []
    lstm2_total_accu = []

    rnn_state = session.run([rnn_model.init_state])
    lstm1_state = session.run([lstm1_model.init_state])
    lstm2_state = session.run([lstm2_model.init_state])

    for step in range(epoch_size):
        lstm1_outputs = []
        for lstm1_slice in range(3):
            rnn_outputs = []
            for rnn_slice in range(6):
                x, y, end = next(data)
                rnn_cost, _, rnn_accuracy, rnn_state, rnn_output = session.run(
                    [
                        rnn_model.cost,
                        rnn_model.cost,
                        rnn_model.accuracy,
                        rnn_model.final_state,
                        rnn_model.output,
                    ],
                    feed_dict={
                        rnn_model.x: x,
                        rnn_model.y: y,
                        rnn_model.init_state: rnn_state
                    }
                )
                rnn_outputs.append(rnn_output)
                rnn_total_accu.append(rnn_accuracy)
                if end is True:
                    rnn_outputs.extend([np.zeros([10, 100])] * (5 - rnn_slice))
                    break
            lstm1_x = np.concatenate(rnn_outputs, axis=0)
            rnn_outputs.clear()
            lstm1_cost, _, lstm1_accuracy, lstm1_state, lstm1_output = session.run(
                [
                    lstm1_model.cost,
                    lstm1_model.optimizer if is_tranning else lstm1_model.cost,
                    lstm1_model.accuracy,
                    lstm1_model.final_state,
                    lstm1_model.output,
                ],
                feed_dict={
                    lstm1_model.x: lstm1_x,
                    lstm1_model.y: y,
                    lstm1_model.init_state: lstm1_state
                }
            )
            lstm1_outputs.append(lstm1_output)
            lstm1_total_accu.append(lstm1_accuracy)
            if end is True:
                lstm1_outputs.extend([np.zeros([60, 150])] * (2 - lstm1_slice))
                break
        lstm2_x = np.concatenate(lstm1_outputs, axis=0)
        lstm1_outputs.clear()
        lstm2_cost, _, lstm2_accuracy, lstm2_state, lstm2_output, merged = session.run(
            [
                lstm2_model.cost,
                lstm2_model.optimizer if is_tranning else lstm2_model.cost,
                lstm2_model.accuracy,
                lstm2_model.final_state,
                lstm2_model.output,
                lstm2_model.merged
            ],
            feed_dict={
                lstm2_model.x: lstm2_x,
                lstm2_model.y: y,
                lstm2_model.init_state: lstm2_state
            }
        )
        lstm2_total_accu.append(lstm2_accuracy)
        if output_log and (step + 1) % 100 == 0:
            with open('./record.txt', 'a') as f:
                f.write(
                    'After %d steps, rnn, lstm1, lstm2 accuracy is %.3f, %.3f, %.3f\n'
                    %
                    (step + 1, rnn_accuracy, lstm1_accuracy, lstm2_accuracy)
                )
            print(
                'After %d steps, rnn, lstm1, lstm2 accuracy is %.3f, %.3f, %.3f\n'
                %
                (step + 1, rnn_accuracy, lstm1_accuracy, lstm2_accuracy)
            )
    rnn_avg_accuracy = np.mean(rnn_total_accu)
    lstm1_avg_accuracy = np.mean(lstm1_total_accu)
    lstm2_avg_accuracy = np.mean(lstm2_total_accu)
    return rnn_avg_accuracy, lstm1_avg_accuracy, lstm2_avg_accuracy, merged


def rnn_run(session, data, is_training, epoch_size, model):

    total_accu = []
    state = session.run([model.init_state])

    for step in range(epoch_size):
        x, y, end = next(data)
        cost, _, accuracy, state, merged = session.run(
            [
                model.cost,
                model.optimizer if is_training else model.cost,
                model.accuracy,
                model.final_state,
                model.merged
            ],
            feed_dict={
                model.x: x,
                model.y: y,
                model.init_state: state
            }
        )

        total_accu.append(accuracy)
        if is_training and (step + 1) % 100 == 0:
            with open('./record.txt', 'a') as f:
                f.write(
                    'After %d steps, rnn accuracy is %.3f\n'
                    %
                    (step + 1, accuracy)
                )
            print(
                'After %d steps, rnn accuracy is %.3f\n'
                %
                (step + 1, accuracy)
            )
    avg_accuracy = np.mean(total_accu)
    return avg_accuracy, merged


def lstm1_run(session, data, is_training, epoch_size, **kwargs):
    rnn_model = kwargs.get('rnn_model')
    lstm1_model = kwargs.get('lstm1_model')

    rnn_total_accu = []
    lstm1_total_accu = []

    rnn_state = session.run([rnn_model.init_state])
    lstm1_state = session.run([lstm1_model.init_state])

    for step in range(epoch_size):
        rnn_outputs = []
        for rnn_slice in range(6):
            x, y, end = next(data)
            rnn_accuracy, rnn_state, rnn_output = session.run(
                [
                    rnn_model.accuracy,
                    rnn_model.final_state,
                    rnn_model.output,
                ],
                feed_dict={
                    rnn_model.x: x,
                    rnn_model.y: y,
                    rnn_model.init_state: rnn_state
                }
            )
            rnn_outputs.append(rnn_output)
            rnn_total_accu.append(rnn_accuracy)
            if end is True:
                rnn_outputs.extend([np.zeros([10, 100])] * (5 - rnn_slice))
                break
        lstm1_x = np.concatenate(rnn_outputs, axis=0)
        rnn_outputs.clear()
        lstm1_cost, _, lstm1_accuracy, lstm1_state, merged = session.run(
            [
                lstm1_model.cost,
                lstm1_model.optimizer,
                lstm1_model.accuracy,
                lstm1_model.final_state,
                lstm1_model.merged
            ],
            feed_dict={
                lstm1_model.x: lstm1_x,
                lstm1_model.y: y,
                lstm1_model.init_state: lstm1_state
            }
        )
        lstm1_total_accu.append(lstm1_accuracy)

        if is_training and (step + 1) % 100 == 0:
            with open('./record.txt', 'a') as f:
                f.write(
                    'After %d steps, rnn, lstm1 accuracy is %.3f, %.3f\n'
                    %
                    (step + 1, rnn_accuracy, lstm1_accuracy)
                )
            print(
                'After %d steps, rnn, lstm1 accuracy is %.3f, %.3f\n'
                %
                (step + 1, rnn_accuracy, lstm1_accuracy)
            )
    rnn_avg_accuracy = np.mean(rnn_total_accu)
    lstm1_avg_accuracy = np.mean(lstm1_total_accu)
    return rnn_avg_accuracy, lstm1_avg_accuracy, merged


def main():
    rnn_train_data = data.np_load(path='G:/sound_fixed', batch_type='train')

    train_data = data.np_load(path='G:/sound_npy', batch_type='train')
    valid_data = data.np_load(path='G:/sound_npy', batch_type='eval')
    test_data = data.np_load(path='G:/sound_npy', batch_type='test')

    train_epoch_size = 1000

    valid_epoch_size = 100

    test_epoch_size = 300

    restore_check_point = True
    check_point_path = './model/sound_test'

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('sound_layers_model', reuse=None, initializer=initializer):
        rnn_train_model = RNNLayer(True, time_slices=10, mfcc_features=512, classes=59)
        lstm1_train_model = LSTM1Layer(True, time_slices=60, mfcc_features=100, classes=59)
        lstm2_train_model = LSTM2Layer(True, time_slices=180, mfcc_features=150, classes=59)

    with tf.variable_scope('sound_layers_model', reuse=True, initializer=initializer):
        rnn_eval_model = RNNLayer(True, time_slices=10, mfcc_features=512, classes=59)
        lstm1_eval_model = LSTM1Layer(True, time_slices=60, mfcc_features=100, classes=59)
        lstm2_eval_model = LSTM2Layer(True, time_slices=180, mfcc_features=150, classes=59)

    saver = tf.train.Saver()

    with tf.Session() as session:
        if restore_check_point and tf.train.checkpoint_exists(check_point_path):
            saver.restore(session, check_point_path)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        writer = tf.summary.FileWriter('.', session.graph)
        rnn_best_accuracy = 0
        lstm1_best_accuracy = 0
        lstm2_best_accuracy = 0
        for i in range(RNN_ITERS):
            rnn_run(session, rnn_train_data, True, train_epoch_size, model=rnn_train_model)

            accuracy, merged = rnn_run(
                session, rnn_train_data, False, train_epoch_size, model=rnn_eval_model
            )

            print('RNN Iter: %d Validation Accuracy: %.3f' % (i, accuracy))
            if rnn_best_accuracy < accuracy:
                saver.save(session, check_point_path)
                rnn_best_accuracy = accuracy

        for i in range(LSTM1_ITERS):
            lstm1_run(session, train_data, True, train_epoch_size, rnn_model=rnn_train_model,
                      lstm1_model=lstm1_train_model)

            rnn_accuracy, lstm1_accuracy, merged = lstm1_run(
                session, rnn_train_data, False, train_epoch_size, rnn_model=rnn_eval_model, lstm1_model=lstm1_eval_model
            )

            print('LSTM1 Iter: %d Validation Accuracy: %.3f, %.3f' % (i, rnn_accuracy, lstm1_accuracy))
            if lstm1_best_accuracy < lstm1_accuracy:
                saver.save(session, check_point_path)
                lstm1_best_accuracy = lstm1_accuracy

        for i in range(NUM_EPOCH):
            with open('./record.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print('In iteration: %d' % (i + 1))
            run_epoch(
                session,
                train_data,
                True,
                True,
                train_epoch_size,
                rnn_model=rnn_eval_model,
                lstm1_model=lstm1_train_model,
                lstm2_model=lstm2_train_model
            )

            rnn_accuracy, lstm1_accuracy, lstm2_accuracy, merged = run_epoch(
                session, valid_data, False, False, valid_epoch_size,
                rnn_model=rnn_eval_model,
                lstm1_model=lstm1_eval_model,
                lstm2_model=lstm2_eval_model
            )
            with open('./record.txt', 'a') as f:
                f.write('In iteration: %d\n' % (i + 1))
            print(
                'Epoch: %d Validation Accuracy: %.3f, %.3f, %.3f'
                %
                (i + 1, rnn_accuracy, lstm1_accuracy, lstm2_accuracy)
            )

            writer.add_summary(merged, i)

            if lstm2_accuracy > lstm2_best_accuracy:
                saver.save(session, check_point_path)
                lstm2_best_accuracy = lstm2_accuracy

        rnn_accuracy, lstm1_accuracy, lstm2_accuracy, merged = run_epoch(
            session, test_data, False, False, test_epoch_size,
            rnn_model=rnn_eval_model,
            lstm1_model=lstm1_eval_model,
            lstm2_model=lstm2_eval_model
        )
        with open('./record.txt', 'a') as f:
            f.write('In iteration: %d\n' % (i + 1))
        print('Test Accuracy: %.3f, %.3f, %.3f' % (rnn_accuracy, lstm1_accuracy, lstm2_accuracy))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()



