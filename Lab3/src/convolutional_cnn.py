import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import numpy as np

batch_size = 30
test_batch_size = 30
batches_count = 1000
tests_count = 100

width = 320
height = 320
channels = 3
hidden_layer1_size = 1000
hidden_layer2_size = 300
classes_size = 3

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch3d(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch3d (test_images, test_labels, test_batch_size)

x = tf.placeholder(tf.float32, [None, width, height, channels])
y = tf.placeholder(tf.float32, [None, classes_size])
W1 = tf.get_variable('W1', [5, 5, 3, 80], initializer=tf.random_normal_initializer())

conv1 = tf.nn.conv2d(x, W1, strides = [1, 1, 1, 1], padding="SAME")
biases1 = tf.get_variable('biases1', [80], initializer=tf.constant_initializer(0.0))
pre_activation1 = tf.nn.bias_add(conv1, biases1)
conv1 = tf.nn.relu(pre_activation1)

pool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

W2 = tf.get_variable('W2', [5, 5, 80, 80], initializer=tf.random_normal_initializer())
conv2 = tf.nn.conv2d(pool1, W2, strides = [1, 1, 1, 1], padding="SAME")
biases2 = tf.get_variable('biases2', [80], initializer=tf.constant_initializer(0.0))
pre_activation2 = tf.nn.bias_add(conv2, biases2)
conv2 = tf.nn.relu(pre_activation2)

pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

reshaped_size = 20 * 20 * 80
shape = tf.stack([-1, reshaped_size])
output = tf.reshape(pool2, shape)

# output = tf.reshape(pool2, shape=[reshaped_size, -1, -1])
W3 = tf.get_variable('W3', [reshaped_size, classes_size], initializer=tf.random_normal_initializer())
biases3 = tf.get_variable('biases3', [classes_size], initializer=tf.constant_initializer(0.0))

y_ = tf.nn.softmax(tf.matmul(output, W3) + biases3)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

# Train
with tf.Session() as sess:
    sess.run (init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for _ in range(batches_count):
        print (100 * _/batches_count,'%\r', end ='')

        x_b, y_b = sess.run ([x_tensor, y_tensor])
        x_b = x_b / 255
        vec_y = np.zeros([batch_size, 3])
        vec_y[range(batch_size), y_b] = 1

        sess.run(train_step, feed_dict={ x: x_b, y: vec_y })

    print ("Testing trained model...")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    sum_accuracy = 0
    for _ in range(tests_count):
        x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
        x_test = x_test / 255.  
        resize_test_y = np.zeros ([len(y_test), 3])
        resize_test_y[range(len(y_test)), y_test] = 1   
        sum_accuracy += sess.run(accuracy, feed_dict={x: x_test, y: resize_test_y})
    print("Accuracy: {:f}".format(sum_accuracy / tests_count)) 
    
    coord.request_stop()
    coord.join(threads)