import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import numpy as np

batch_size = 30
test_batch_size = 200
batches_count = 1000
tests_count = 10

features_size = 320 * 320 * 3
hidden_layer_size = 100
classes_size = 3

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch (test_images, test_labels, test_batch_size)

                                #CountOfImages, height, weight, RGB
x = tf.placeholder(tf.float32,  [None, features_size])

y = tf.placeholder(tf.float32, [None, classes_size])

W1 = tf.get_variable('W1', [features_size, hidden_layer_size], initializer=tf.random_normal_initializer())
b1 = tf.get_variable('b1', [1,], initializer=tf.constant_initializer(0.0))
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.get_variable('W2', [hidden_layer_size, classes_size], initializer=tf.random_normal_initializer())
b2 = tf.get_variable('b2', [1,], initializer=tf.constant_initializer(0.0))

y_ = tf.nn.softmax(tf.matmul(y1, W2) + b2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cross_entropy)
init_op = tf.initialize_all_variables()

# Train
with tf.Session() as sess:
    sess.run (init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for _ in range(batches_count):
        # print (_)
        # print ("Batch..")
        print (100 * _/batches_count,'%\r', end ='')
        x_b, y_b = sess.run ([x_tensor, y_tensor])
        x_b = x_b / 255
        vec_y = np.zeros([batch_size, 3])
        vec_y[range(batch_size), y_b] = 1
        # print ("X:")    
        # print (x_b)
        # print ("Y:")
        # print (y_b)
        # print (vec)
        # print ("Feeding...")
        sess.run(train_step, feed_dict={ x: x_b, y: vec_y })

    coord.request_stop()
    coord.join(threads)

    print ("Testing trained model...")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
    x_test = x_test / 255.  
    resize_test_y = np.zeros ([len(y_test), 3])
    resize_test_y[range(len(y_test)), y_test] = 1   
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    print(sess.run(accuracy, feed_dict={x: x_test, y: resize_test_y}))  
    
    coord.request_stop()
    coord.join(threads)
