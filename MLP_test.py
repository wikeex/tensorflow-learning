# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("./MNIST/MNIST-data", one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 100
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# 隐层使用relu激活函数，输出没有负数，tf.sign取输出符号，最后active_unit中为1处则该神经元激活
active_unit = tf.sign(hidden1)

# active_unit取反
inactive_unit = tf.negative(active_unit)

# 这步计算之后，forget_vector向量中激活的神经元为1，未激活的神经元为 0.8
forget_vector = tf.add((inactive_unit + tf.ones([100, 100]))*0, active_unit*0)

# forget_vector切片之后表示每一次输入的激活状态，与W1矩阵点乘，未激活神经元的所有权重乘0.8，压制未激活的特征
vectors = tf.split(forget_vector, 100)
for vector in vectors:
    W1 = W1 * vector

y = tf.nn.softmax(tf.matmul(hidden1, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

W1 = W1 * 0
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        print("step %d, training accuracy %g"%(i, accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

#print('mean:', tf.reduce_mean(W1).eval())
#print('max:', tf.reduce_max(W1).eval())
#print('min:', tf.reduce_min(W1).eval())
#print(tf.reshape(W1, [-1, 28, 28]).eval())

print("final accuracy %g" % (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))
#print(forget_matrix)
