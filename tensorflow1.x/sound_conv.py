import tensorflow as tf
from sound_lstm_test import data

batch_size = 10

x = tf.placeholder(tf.float32, [batch_size, 512, 80])
y_ = tf.placeholder(tf.float32, [batch_size, 59])

w_conv1 = tf.Variable(tf.truncated_normal([16, 2, 1, 64], stddev=0.1), name='conv1_w')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]), name='conv1_b')

x_image = tf.reshape(x, [-1, 512, 80, 1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 2, 1, 1], padding='VALID') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

w_conv2 = tf.Variable(tf.truncated_normal([2, 16, 64, 128], stddev=0.1), name='conv2_w')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]), name='conv2_b')

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

w_fc1 = tf.Variable(tf.truncated_normal([61 * 12 * 128, 1024], stddev=0.1), name='fc1_w')
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='fc1_b')

h_pool2_flat = tf.reshape(h_pool2, [-1, 61 * 12 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

rate = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, rate=rate)

w_fc2 = tf.Variable(tf.truncated_normal([1024, 59], stddev=0.1), name='fc2_w')
b_fc2 = tf.Variable(tf.constant(0.1, shape=[59]), name='fc2_b')

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

variables = tf.trainable_variables()
conv1_variable = [t for t in variables if t.name.startswith('conv1')]
conv2_variable = [t for t in variables if t.name.startswith('conv2')]
fc1_variable = [t for t in variables if t.name.startswith('fc1')]
fc2_variable = [t for t in variables if t.name.startswith('fc2')]

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
grads_conv1, _ = tf.clip_by_global_norm(tf.gradients(loss, conv1_variable), clip_norm=5)
grads_conv2, _ = tf.clip_by_global_norm(tf.gradients(loss, conv2_variable), clip_norm=5)
grads_fc1, _ = tf.clip_by_global_norm(tf.gradients(loss, fc1_variable), clip_norm=5)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, variables), clip_norm=5)

conv1_optimizer = tf.train.AdamOptimizer(0.001)
conv2_optimizer = tf.train.AdamOptimizer(0.001)
fc1_optimizer = tf.train.AdamOptimizer(0.001)
fc2_optimizer = tf.train.AdamOptimizer(0.001)
optimizer = tf.train.AdamOptimizer(0.001)

conv1_op = conv1_optimizer.apply_gradients(zip(grads_conv1, conv1_variable))
conv2_op = conv2_optimizer.apply_gradients(zip(grads_conv2, conv2_variable))
fc1_op = fc1_optimizer.apply_gradients(zip(grads_fc1, fc1_variable))
fc2_op = fc2_optimizer.apply_gradients(zip(grads_fc2, fc2_variable))
op = optimizer.apply_gradients(zip(grads, variables))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_data = data.np_load(batch_size=10, batch_type='train/')
    test_data = data.np_load(batch_size=10, batch_type='test/')

    for i in range(1000):
        for _ in range(100):
            input_, label = next(train_data)
            sess.run([conv1_op, conv2_op, fc1_op, fc2_op], feed_dict={x: input_, y_: label, rate: 0})

        test_total_accuracy = 0
        for i in range(10):
            test_input_, test_label = next(test_data)
            test_accuracy, _ = sess.run([accuracy, tf.no_op()], feed_dict={x: test_input_, y_: test_label, rate: 0})
            test_total_accuracy += test_accuracy
        print('测试集准确度：%.3f' % (test_total_accuracy / 10))
