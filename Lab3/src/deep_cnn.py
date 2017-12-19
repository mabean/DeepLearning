import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import numpy as np

def batch_norm(x, shape, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[shape]),
                                         name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

batch_size = 5
test_batch_size = 5
batches_count = 10000
tests_count = 500

width = 256
height = 256
channels = 3
classes_size = 3

phase_train = tf.placeholder(tf.bool, name='phase_train')
filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch3d(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch3d (test_images, test_labels, test_batch_size)

x = tf.placeholder(tf.float32, [None, width, height, channels])
y = tf.placeholder(tf.float32, [None, classes_size])

W1 = tf.Variable(tf.random_normal([5, 5, 3, 64]))
conv1 = tf.nn.conv2d(x, W1, strides = [1, 1, 1, 1], padding="SAME")
biases1 = tf.Variable(tf.zeros([64]))
pre_activation1 = tf.nn.bias_add(conv1, biases1)
norm1 = batch_norm(pre_activation1, 64, phase_train)
conv1 = tf.nn.relu(norm1)

pool1 = tf.nn.avg_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([5, 5, 64, 128]))
conv2 = tf.nn.conv2d(pool1, W2, strides = [1, 1, 1, 1], padding="SAME")
biases2 = tf.Variable(tf.zeros([128]))
pre_activation2 = tf.nn.bias_add(conv2, biases2)
norm2 = batch_norm(pre_activation2, 128, phase_train)
conv2 = tf.nn.relu(norm2)

pool2 = tf.nn.avg_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([5, 5, 128, 128]))
conv3 = tf.nn.conv2d(pool2, W3, strides = [1, 1, 1, 1], padding="SAME")
biases3 = tf.Variable(tf.zeros([128]))
pre_activation3 = tf.nn.bias_add(conv3, biases3)
norm3 = batch_norm(pre_activation3, 128, phase_train)
conv3 = tf.nn.relu(norm3)

pool3 = tf.nn.avg_pool(conv3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

reshaped_size = 16 * 16 * 128
shape = tf.stack([-1, reshaped_size])
dropout = tf.reshape(pool3, shape)

# output = tf.reshape(pool2, shape=[reshaped_size, -1, -1])
W4 = tf.Variable(tf.random_normal([reshaped_size, classes_size]))
biases4 = tf.Variable(tf.zeros([classes_size]))

y_ = tf.nn.softmax(tf.matmul(dropout, W4) + biases4)
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

        sess.run(train_step, feed_dict={ x: x_b, y: vec_y , phase_train: True})

        if (_ % 1000 == 0 and _ > 0):
            print("", end = '')
            print("Test on {:n}".format(_))
            sum_accuracy = 0
            for t in range(tests_count):
                print("Testing: ", end='')
                print (100 * t / tests_count,'%\r', end ='')
                x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
                x_test = x_test / 255.  
                resize_test_y = np.zeros ([len(y_test), 3])
                resize_test_y[range(len(y_test)), y_test] = 1   
                sum_accuracy += sess.run(accuracy, feed_dict={x: x_test, y: resize_test_y, phase_train: False})
            print("Accuracy: {:f}".format(sum_accuracy / tests_count)) 
    
    coord.request_stop()
    coord.join(threads)