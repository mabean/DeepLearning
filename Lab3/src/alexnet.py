import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import numpy as np

batch_size = 5
test_batch_size = 5
batches_count = 50000
tests_count = 500

width = 256
height = 256
channels = 3
classes_size = 3

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch3d(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch3d (test_images, test_labels, test_batch_size)

x = tf.placeholder(tf.float32, [None, width, height, channels])
y = tf.placeholder(tf.float32, [None, classes_size])

with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(x, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)

# lrn1
with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

pool1 = tf.nn.max_pool(lrn1,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID',
                        name='pool1')

with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)

# lrn2
with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                            alpha=1e-4,
                                            beta=0.75,
                                            depth_radius=2,
                                            bias=2.0)
# pool2
pool2 = tf.nn.max_pool(lrn2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool2')

# conv3
with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)

# conv4
with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)

# conv5
with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)

# pool5
pool5 = tf.nn.avg_pool(conv5,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool5')


dropout = tf.reshape(pool5, [-1, 62720])
W = tf.Variable(tf.truncated_normal([62720, classes_size],
                                    dtype=tf.float32,
                                    stddev=1e-1), name='weights')
b = tf.Variable(tf.constant(0.0, shape=[classes_size], dtype=tf.float32),
                         trainable=True, name='biases')

y_ = tf.nn.softmax(tf.matmul(dropout, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

# Train
with tf.Session() as sess:
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

        sess.run(train_step, feed_dict={ x: x_b, y: vec_y })

        if (_ % 1000 == 0 and _ > 0):
            print("Test on {:n}".format(_))
            sum_accuracy = 0
            for t in range(tests_count):
                print("Testing: ", end='')
                print (100 * t / tests_count,'%\r', end ='')
                x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
                x_test = x_test / 255.  
                resize_test_y = np.zeros ([len(y_test), 3])
                resize_test_y[range(len(y_test)), y_test] = 1   
                sum_accuracy += sess.run(accuracy, feed_dict={x: x_test, y: resize_test_y })
            print("Accuracy: {:f}".format(sum_accuracy / tests_count)) 
    
    coord.request_stop()
    coord.join(threads)