import tensorflow as tf
import numpy as np
from SoundLayers import real_time_data, data
from SoundLayers.model import RNNLayer, LSTM1Layer, LSTM2Layer


def lstm2_run(session, data_iter, **kwargs):

    rnn_model = kwargs.get('rnn_model')
    lstm1_model = kwargs.get('lstm1_model')
    lstm2_model = kwargs.get('lstm2_model')

    rnn_state = session.run([rnn_model.init_state])
    lstm1_state = session.run([lstm1_model.init_state])
    lstm2_state = session.run([lstm2_model.init_state])

    one_hot = data.one_hot_from_files()

    while True:
        lstm1_outputs = []
        for lstm1_slice in range(2):
            rnn_outputs = []
            for rnn_slice in range(6):
                x = next(data_iter)
                y = np.zeros((59,), dtype=np.float32)
                rnn_state, rnn_output, rnn_result = session.run(
                    [
                        rnn_model.final_state,
                        rnn_model.output,
                        rnn_model.logits_softmax
                    ],
                    feed_dict={
                        rnn_model.x: x,
                        rnn_model.y: y,
                        rnn_model.init_state: rnn_state
                    }
                )
                rnn_outputs.append(rnn_output)
                classification = data.extract_classification(rnn_result, one_hot)
                print(classification)
            lstm1_x = np.concatenate(rnn_outputs, axis=0)
            rnn_outputs.clear()
            lstm1_state, lstm1_output, lstm1_result = session.run(
                [
                    lstm1_model.final_state,
                    lstm1_model.output,
                    lstm1_model.logits_softmax
                ],
                feed_dict={
                    lstm1_model.x: lstm1_x,
                    lstm1_model.y: y,
                    lstm1_model.init_state: lstm1_state
                }
            )
            lstm1_outputs.append(lstm1_output)
            classification = data.extract_classification(lstm1_result, one_hot)
            print(classification)
        lstm2_x = np.concatenate(lstm1_outputs, axis=0)
        lstm1_outputs.clear()
        lstm2_state, lstm2_output, lstm2_result = session.run(
            [
                lstm2_model.final_state,
                lstm2_model.output,
                lstm2_model.logits_softmax
            ],
            feed_dict={
                lstm2_model.x: lstm2_x,
                lstm2_model.y: y,
                lstm2_model.init_state: lstm2_state
            }
        )
        classification = data.extract_classification(lstm2_result, one_hot)
        print(classification)


def main():

    data = real_time_data.real_time_sound()

    restore_check_point = True
    check_point_path = './model/sound_test'

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('sound_layers_model', reuse=None, initializer=initializer):
        rnn_train_model = RNNLayer(True, time_slices=10, mfcc_features=512, classes=59)
        lstm1_train_model = LSTM1Layer(True, time_slices=60, mfcc_features=100, classes=59)
        lstm2_train_model = LSTM2Layer(True, time_slices=120, mfcc_features=150, classes=59)

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

        lstm2_run(
            session, data,
            rnn_model=rnn_eval_model,
            lstm1_model=lstm1_eval_model,
            lstm2_model=lstm2_eval_model
        )


if __name__ == '__main__':
    main()

