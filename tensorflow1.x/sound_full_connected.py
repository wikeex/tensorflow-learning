import tensorflow as tf
from sound_lstm_test import data

L1_HIDDEN_SIZE = 400
L2_HIDDEN_SIZE = 200
batch_size = 10
epoch_size = 1000

train_data = data.np_load(batch_size=10, batch_type='train/')
test_data = data.np_load(batch_size=10, batch_type='test/')

x = tf.placeholder(tf.float32, [batch_size, 512, 80])
y_ = tf.placeholder(tf.float32, [batch_size, 59])

x_shaped = tf.reshape(x, shape=[-1, 40960])

w1 = tf.Variable(tf.truncated_normal([40960, L1_HIDDEN_SIZE], stddev=0.1))
b1 = tf.Variable(tf.zeros([L1_HIDDEN_SIZE]))
#w2 = tf.Variable(tf.truncated_normal([L1_HIDDEN_SIZE, L2_HIDDEN_SIZE], stddev=0.1))
#b2 = tf.Variable(tf.zeros([L2_HIDDEN_SIZE]))
w3 = tf.Variable(tf.truncated_normal([L1_HIDDEN_SIZE, 59], stddev=0.1))
b3 = tf.Variable(tf.zeros([59]))

hidden1 = tf.nn.relu(tf.matmul(x_shaped, w1) + b1)
#hidden2 = tf.nn.softmax(tf.matmul(hidden1, w2) + b2)

y = tf.matmul(hidden1, w3) + b3

correct_prediction = tf.equal(tf.arg_max(tf.nn.softmax(y), 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_, y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()

    for i in range(100):
        for _ in range(epoch_size):
            input_x, input_y = next(train_data)
            _ = sess.run([optimizer, ], feed_dict={x: input_x, y_: input_y})

        test_total_accuracy = 0
        for j in range(10):
            test_input_, test_label = next(test_data)
            _, test_accuracy = sess.run([tf.no_op(), accuracy], feed_dict={x: test_input_, y_: test_label})
            test_total_accuracy += test_accuracy
        print('测试准确度：{:.3f}%'.format(test_total_accuracy / 10))
