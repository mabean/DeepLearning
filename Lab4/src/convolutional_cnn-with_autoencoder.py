import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
import tensorflow as tf
import numpy as np

batch_size = 5
test_batch_size = 5
batches_count = 10000
tests_count = 1000

num_steps = 10000
width = 256
height = 256
channels = 3
classes_size = 3
learning_rate = 0.01
display_step = 1000

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch3d(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch3d (test_images, test_labels, test_batch_size)

x = tf.placeholder(tf.float32, [None, width, height, channels])
y = tf.placeholder(tf.float32, [None, classes_size])
W1_encoder = tf.get_variable('W1_encoder', [5, 5, 3, 64], initializer=tf.random_normal_initializer())
W2_encoder = tf.get_variable('W2_encoder', [5, 5, 64, 64], initializer=tf.random_normal_initializer())

W1_decoder = tf.get_variable('W1_decoder', [5, 5, 64, 64], initializer=tf.random_normal_initializer())
W2_decoder = tf.get_variable('W2_decoder', [5, 5, 3, 64], initializer=tf.random_normal_initializer())

def encoder(X):
    conv1 = tf.nn.conv2d(X, W1_encoder, strides = [1, 1, 1, 1], padding="SAME")
    biases1 = tf.get_variable('biases1', [64], initializer=tf.constant_initializer(0.0))
    pre_activation1 = tf.nn.bias_add(conv1, biases1)
    conv1 = tf.nn.relu(pre_activation1)

    conv2 = tf.nn.conv2d(conv1, W2_encoder, strides = [1, 1, 1, 1], padding="SAME")
    biases2 = tf.get_variable('biases2', [64], initializer=tf.constant_initializer(0.0))
    pre_activation2 = tf.nn.bias_add(conv2, biases2)
    conv2 = tf.nn.relu(pre_activation2)
    return conv2

def decoder(X_):
    conv3 = tf.nn.conv2d_transpose (X_, W1_decoder,
        tf.stack([5, 5, 64, 64]), strides = [1, 1, 1, 1], padding="SAME")
    biases_decoder1 = tf.get_variable('biases_decoder1', [64], initializer=tf.constant_initializer(0.0))
    pre_activation3 = tf.nn.bias_add(conv3, biases_decoder1)
    conv3 = tf.nn.relu(pre_activation3)

    conv4 = tf.nn.conv2d_transpose(X_, W2_decoder,
        tf.stack([tf.shape(x)[0], 256, 256, 3]), strides = [1, 1, 1, 1], padding="SAME")

    #biases_decoder2 = tf.Variable(tf.zeros([conv4.get_shape().as_list()[3]]))
    biases_decoder2 = tf.get_variable('biases_decoder2', [3], initializer=tf.constant_initializer(0.0))
    pre_activation4 = tf.nn.bias_add(conv4, biases_decoder2)
    conv4 = tf.nn.relu(pre_activation4)
    return conv4

encoder_op = encoder(x)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = x

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

reshaped_size = 256 * 256 * 64
shape = tf.stack([-1, reshaped_size])
output = tf.reshape(encoder_op, shape)

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

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

    print ("Autoencoder...")
    for i in range(1, num_steps+1):
        print (100 * i/num_steps,'%\r', end ='')
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = sess.run ([x_tensor, y_tensor])
        batch_x = batch_x / 255

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={x: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

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
            for _ in range(tests_count):
                x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
                x_test = x_test / 255.  
                resize_test_y = np.zeros ([len(y_test), 3])
                resize_test_y[range(len(y_test)), y_test] = 1   
                sum_accuracy += sess.run(accuracy, feed_dict={x: x_test, y: resize_test_y})
            print("Accuracy: {:f}".format(sum_accuracy / tests_count)) 
    
    coord.request_stop()
    coord.join(threads)
