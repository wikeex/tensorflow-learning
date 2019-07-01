import tensorflow as tf
import numpy as np
from SoundLayers import data
from SoundLayers.model import RNNLayer, LSTM1Layer, LSTM2Layer


def lstm2_run(session, data, **kwargs):

    rnn_model = kwargs.get('rnn_model')
    lstm1_model = kwargs.get('lstm1_model')
    lstm2_model = kwargs.get('lstm2_model')

    rnn_state = session.run([rnn_model.init_state])
    lstm1_state = session.run([lstm1_model.init_state])
    lstm2_state = session.run([lstm2_model.init_state])

    lstm1_outputs = []
    for lstm1_slice in range(2):
        rnn_outputs = []
        for rnn_slice in range(6):
            x = next(data)
            rnn_accuracy, rnn_state, rnn_output = session.run(
                [
                    rnn_model.accuracy,
                    rnn_model.final_state,
                    rnn_model.output,
                ],
                feed_dict={
                    rnn_model.x: x,
                    rnn_model.init_state: rnn_state
                }
            )
            rnn_outputs.append(rnn_output)

        lstm1_x = np.concatenate(rnn_outputs, axis=0)
        rnn_outputs.clear()
        lstm1_accuracy, lstm1_state, lstm1_output = session.run(
            [
                lstm1_model.accuracy,
                lstm1_model.final_state,
                lstm1_model.output,
            ],
            feed_dict={
                lstm1_model.x: lstm1_x,
                lstm1_model.init_state: lstm1_state
            }
        )
        lstm1_outputs.append(lstm1_output)

        lstm2_x = np.concatenate(lstm1_outputs, axis=0)
        lstm1_outputs.clear()
        lstm2_accuracy, lstm2_state = session.run(
            [
                lstm2_model.accuracy,
                lstm2_model.final_state,
            ],
            feed_dict={
                lstm2_model.x: lstm2_x,
                lstm2_model.init_state: lstm2_state
            }
        )

    return


def main():

    batch = data.real_time_sound()

    valid_epoch_size = 100

    restore_check_point = True
    check_point_path = './model/sound_test'

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope('sound_layers_model', reuse=True, initializer=initializer):
        rnn_eval_model = RNNLayer(False, time_slices=10, mfcc_features=512, classes=59)
        lstm1_eval_model = LSTM1Layer(False, time_slices=60, mfcc_features=100, classes=59)
        lstm2_eval_model = LSTM2Layer(False, time_slices=120, mfcc_features=150, classes=59)

    saver = tf.train.Saver()

    with tf.Session() as session:
        if restore_check_point and tf.train.checkpoint_exists(check_point_path):
            saver.restore(session, check_point_path)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        while True:

            lstm2_run(
                session, batch, False,
                rnn_model=rnn_eval_model,
                lstm1_model=lstm1_eval_model,
                lstm2_model=lstm2_eval_model
            )

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()



