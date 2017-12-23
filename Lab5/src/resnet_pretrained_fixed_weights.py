import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2, resnet_utils
import numpy as np

height = 256
width = 256
channels = 3

batch_size = 5
test_batch_size = 5
batches_count = 10000
tests_count = 500

classes_size = 3

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch3d(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch3d (test_images, test_labels, test_batch_size)

# Create graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
y = tf.placeholder(tf.float32, [None, 3])

with slim.arg_scope(resnet_utils.resnet_arg_scope()):
    logits, end_points = resnet_v2.resnet_v2_50(X, is_training=True)

shape = tf.stack([-1, 2048])
dropout = tf.reshape(logits, shape)

W = tf.Variable(tf.random_normal([2048, classes_size]))
biases = tf.Variable(tf.constant(0.0, shape=[classes_size], dtype=tf.float32),
                         trainable=True, name='biases')

y_ = tf.nn.softmax(tf.matmul(dropout, W) + biases)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

# Execute graph
with tf.Session() as sess:
    
    saver.restore(sess, "../models/resnet_v2_152_2017_04_14/resnet_v2_152.ckpt")
    sess.run (init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

    for _ in range(batches_count + 1):
        print (100 * _/batches_count,'%\r', end ='')

        x_b, y_b = sess.run ([x_tensor, y_tensor])
        x_b = x_b / 255
        vec_y = np.zeros([batch_size, 3])
        vec_y[range(batch_size), y_b] = 1

        sess.run(train_step, feed_dict={ X: x_b, y: vec_y })

        if (_ % 1000 == 0 and _ > 0):
            print("", end = '')
            print("Test on {:n}".format(_))
            sum_accuracy = 0
            for _ in range(tests_count):
                print("Testing: ", end='')
                print (100 * t / tests_count,'%\r', end ='')
                x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
                x_test = x_test / 255.  
                resize_test_y = np.zeros ([len(y_test), 3])
                resize_test_y[range(len(y_test)), y_test] = 1   
                sum_accuracy += sess.run(accuracy, feed_dict={X: x_test, y: resize_test_y})
            print("Accuracy: {:f}".format(sum_accuracy / tests_count)) 
    
    coord.request_stop()
    coord.join(threads)