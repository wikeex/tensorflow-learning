import numpy as np
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./datasets/MNIST/MNIST-data')

x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu
}
conv1 = tf.layers.conv2d(x, name='conv1', **conv1_params)

caps1_num = 32
caps1_dims = 8
conv2_params = {
    "filters": caps1_num * caps1_dims,
    'kernel_size': 9,
    'strides': 2,
    'padding': 'valid',
    'activation': tf.nn.relu
}
conv2 = tf.layers.conv2d(conv1, name='conv2', **conv2_params)

caps1_caps = caps1_num * 6 * 6
caps1_raw = tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name='caps1_raw')


def squash(s, axis=-1, epsilon=1e-7):
    s_sqr_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    V = s_sqr_norm/(1. + s_sqr_norm) / tf.sqrt(s_sqr_norm + epsilon)
    return V * s


caps1_output = squash(caps1_raw)

caps2_caps = 10
caps2_dims = 16

init_sigma = 0.01

W_init = tf.random_normal(
    shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims),
    stddev=init_sigma, dtype=tf.float32, name='W_init'
)
W = tf.Variable(W_init, name='W')

batch_size = tf.shape(x)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name='W_tiled')

caps1_output_expanded = tf.expand_dims(caps1_output, -1, name='caps1_output_expanded')
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name='caps1_output_tile')
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_caps, 1, 1], name='caps1_output_tiled')

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name='caps2_predicted')

# 动态路由算法
raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=tf.float32, name='raw_weights')
b = raw_weights
routing_num = 2
for i in range(routing_num):
    c = tf.nn.softmax(b, axis=2)
    preds = tf.multiply(c, caps2_predicted)
    s = tf.reduce_sum(preds, axis=1, keepdims=True)
    vj = squash(s, axis=-2)

    if i < routing_num - 1:
        vj_tiled = tf.tile(vj, [1, caps1_caps, 1, 1, 1], name='vj_tiled')
        agreement = tf.matmul(caps2_predicted, vj_tiled, transpose_a=True, name='agreement')
        b += agreement
caps2_output = vj


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name='safe_norm'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


y_proba = safe_norm(vj, axis=-2, name='y_proba')
y_proba_argmax = tf.argmax(y_proba, axis=2, name='y_proba')
y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name='y_pred')

y = tf.placeholder(shape=[None], dtype=tf.int64)

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_caps, name='T')

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name='caps2_output_norm')
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name='present_error_raw')
present_error = tf.reshape(present_error_raw, shape=(-1, 10), name='absent_error')
absent_error = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name='absent_error')
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name='L')
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name='margin_loss')

mask_with_labels = tf.placeholder_with_default(False, shape=(), name='mask_with_labels')
reconstruction_targets = tf.cond(mask_with_labels, lambda: y, lambda: y_pred, name='reconstruction_targets')
reconstruction_mask = tf.one_hot(reconstruction_targets, depth=caps2_caps, name='reconstruction_mask')

reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_caps, 1, 1],
    name='reconstruction_mask_reshaped'
)
caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name='caps2_output_masked'
)

n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28

decoder_input = tf.reshape(caps2_output_masked, [-1, caps2_caps * caps2_dims], name='decoder_input')
with tf.name_scope('decoder'):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name='hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2')
    decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name='decoder_output')

x_flat = tf.reshape(x, [-1, n_output], name='x_flat')
squared_difference = tf.square(x_flat - decoder_output, name='squard_difference')
reconstruction_loss = tf.reduce_sum(squared_difference, name='reconstruction_loss')

alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name='loss')

correct = tf.equal(y, y_pred, name='correct')
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name='training_op')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epochs = 5
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = mnist.train.num_examples
n_iterations_validation = mnist.validation.num_examples
best_loss_val = np.infty
checkpoint_path = './my_capsule_network'

training_start_time = time.time()

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={
                    x: x_batch.reshape([-1, 28, 28, 1]),
                    y: y_batch,
                    mask_with_labels: True
                }
            )
            print('\rIteration: {}/{} ({:.1f}%) Loss: {:.5f} time: {}s'.format(
                iteration, n_iterations_per_epoch, iteration * 100 / n_iterations_per_epoch,
                loss_train, int(time.time() - training_start_time)
            ), end='')

        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            x_batch, y_batch = mnist.validation.next_batch(batch_size)
            loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={
                    x: x_batch.reshape([-1, 28, 28, 1]),
                    y: y_batch
                }
            )
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print('\rEvaluation the model: {}/{} ({:.1f}%)'.format(
                iteration, n_iterations_validation,
                iteration * 100 / n_iterations_validation
            ), end=' '*10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print('\rEpoch: {} Val accuracy: {:.4f}% Loss: {:.6f}{}'.format(
            epoch + 1, acc_val * 100, loss_val, '(improved)' if loss_val < best_loss_val else ''
        ))

if loss_val < best_loss_val:
    save_path = saver.save(sess, checkpoint_path)
    best_loss_val = loss_val

n_iterations_test = mnist.test.num_examples

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        x_batch, y_batch = mnist.test.next_batch(batch_size)
        loss_test, acc_test = sess.run(
            [loss, accuracy],
            feed_dict={
                x: x_batch.reshape([-1, 28, 28, 1]),
                y: y_batch
            }
        )
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print('\rEvaluating the model: {}/{} ({:.1f}%)'.format(
            iteration, n_iterations_test, iteration * 100 / n_iterations_test
        ), end=' '*10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print('\rFinal test accuracy: {:.4f}% Loss: {:.6f}'.format(
        acc_test * 100, loss_test
    ))

training_end_time = time.time()
training_time = training_end_time - training_start_time
print('training time: {}'.format(training_time))

