import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import numpy as np

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

batch_size = 30
train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)

                                #CountOfImages, height, weight, RGB
x = tf.placeholder(tf.float32,  [None, 320 * 320 * 3])

y_ = tf.placeholder(tf.float32, [None, 3])

W = tf.get_variable('W', [320 * 320 * 3, 3], initializer=tf.random_normal_initializer())
b = tf.get_variable('b', [1,], initializer=tf.constant_initializer(0.0))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cross_entropy)
init_op = tf.initialize_all_variables()

# Train
with tf.Session() as sess:
    sess.run (init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for _ in range(10):
        # print (_)
        # print ("Batch..")
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
        sess.run(train_step, feed_dict={ x: x_b, y_: vec_y })

    coord.request_stop()
    coord.join(threads)


    # Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: test_images,
#                                     y_: test_labels}))
    