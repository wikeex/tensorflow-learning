import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data

tfe.enable_eager_execution()

with tf.device("/cpu:0"):

    mnist = input_data.read_data_sets("./datasets/MNIST/MNIST-data", one_hot=True)

    x, y = mnist.train.next_batch(100)

    with tf.GradientTape() as tape:
        w = tfe.Variable(tf.zeros([784, 10]))
        b = tfe.Variable(tf.zeros([10]))

        logits = tf.nn.softmax(tf.matmul(x, w) + b)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
        grads = tape.gradient(loss, [w, b])

    optimizer = tf.train.AdamOptimizer(0.0001)

    for i in range(1000):
        optimizer.apply_gradients(zip(grads, [w, b]), global_step=tf.train.get_or_create_global_step())
        print(loss)
