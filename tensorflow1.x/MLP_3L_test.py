# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("./MNIST/MNIST-data", one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 200
h2_units = 200

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
b2 = tf.Variable(tf.zeros([h2_units]))
W3 = tf.Variable(tf.zeros([h2_units, 10]))
b3 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
y = tf.nn.softmax(tf.matmul(hidden2, W3) + b3)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        print("step %d, training accuracy %g" % (i, accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))
    train_step.run({x: batch_xs, y_: batch_ys})

print("final accuracy %g" % (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))
