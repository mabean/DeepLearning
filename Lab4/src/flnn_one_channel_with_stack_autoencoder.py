import sys
sys.path.append('../../data/')
import tfrecords_reader as reader
from tfrecords_converter import rgb_to_y
import tensorflow as tf
import numpy as np

batch_size = 30
test_batch_size = 30
batches_count = 1000
tests_count = 100

features_size = 256 * 256
hidden_layer1_size = 1000
hidden_layer2_size = 300
classes_size = 3
learning_rate = 0.01
num_steps = 1000
display_step = 50

filename_train_queue = tf.train.string_input_producer(["../../data/dataset_train.tfrecords"])
filename_test_queue = tf.train.string_input_producer(["../../data/dataset_test.tfrecords"])

train_images, train_labels = reader.read (filename_train_queue)
x_tensor, y_tensor = reader.next_batch3d(train_images, train_labels, batch_size)

test_images, test_labels = reader.read (filename_test_queue)
x_test_Tensor, y_test_Tensor = reader.next_batch3d(test_images, test_labels, test_batch_size)

                                #CountOfImages, height, weight, RGB
x = tf.placeholder(tf.float32, [None, features_size])

y = tf.placeholder(tf.float32, [None, classes_size])

b1_encoder = tf.get_variable('b1_encoder', [1,], initializer=tf.constant_initializer(0.0))
W1_encoder = tf.get_variable('W1_encoder', [features_size, hidden_layer1_size], initializer=tf.random_normal_initializer())
W1_decoder = tf.get_variable('W1_decoder', [hidden_layer1_size, features_size], initializer=tf.random_normal_initializer())

b2_encoder = tf.get_variable('b2_encoder', [1,], initializer=tf.constant_initializer(0.0))
W2_encoder = tf.get_variable('W2_encoder', [hidden_layer1_size, hidden_layer2_size], initializer=tf.random_normal_initializer())
W2_decoder = tf.get_variable('W2_decoder', [hidden_layer2_size, hidden_layer1_size], initializer=tf.random_normal_initializer())

def encoder_1(X, w, b):
	return tf.nn.relu(tf.matmul(X, w) + b)

def decoder_1(X_):	
	b1_decoder = tf.get_variable('b1_decoder', [1,], initializer=tf.constant_initializer(0.0))
	return tf.nn.relu(tf.matmul(X_, W1_decoder) + b1_decoder)

def encoder_2(X, w, b):
	return tf.nn.relu(tf.matmul(X, w) + b)	

def decoder_2(X_):
	b2_decoder = tf.get_variable('b2_decoder', [1,], initializer=tf.constant_initializer(0.0))
	return tf.nn.relu(tf.matmul(X_, W2_decoder) + b2_decoder)

encoder_op_1 = encoder_1(x, W1_encoder, b1_encoder)
decoder_op_1 = decoder_1(encoder_op_1)

y_pred_1 = decoder_op_1
y_true_1 = x

loss_1 = tf.reduce_mean(tf.pow(y_true_1 - y_pred_1, 2))
optimizer_1 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_1)

W1_encoder.trainable=False
b1_encoder.trainable=False

encoder_op_2 = encoder_2(encoder_op_1, W2_encoder, b2_encoder)
decoder_op_2 = decoder_2(encoder_op_2)

y_pred_2 = decoder_op_2
y_true_2 = encoder_op_1

loss_2 = tf.reduce_mean(tf.pow(y_true_2 - y_pred_2, 2))
optimizer_2 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_2)

W1_encoder.trainable=True
b1_encoder.trainable=True

encoder_op_1 = encoder_1(x, W1_encoder, b1_encoder)
encoder_op_2 = encoder_2(encoder_op_1, W2_encoder, b2_encoder)

W3 = tf.get_variable('W3', [hidden_layer2_size, classes_size], initializer=tf.random_normal_initializer())
b3 = tf.get_variable('b3', [1,], initializer=tf.constant_initializer(0.0))
y_ = tf.nn.softmax(tf.matmul(encoder_op_2, W3) + b3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_),
                                reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

# Train
with tf.Session() as sess:
    sess.run (init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
			
    print ("Autoencoder...2")
    for i in range(1, num_steps+1):
        print (100 * i/num_steps,'%\r', end ='')
        # Prepare Data
        batch_x, _ = sess.run ([x_tensor, y_tensor])
        x_2d_a = rgb_to_y(batch_x[:,:,:,0], batch_x[:,:,:,1], batch_x[:,:,:,2])
        x_b_a = x_2d_a.reshape(batch_size, -1) / 255
    
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer_2, loss_2], feed_dict={x: x_b_a})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
			
    print ("Autoencoder...1")
    for i in range(1, num_steps+1):
        print (100 * i/num_steps,'%\r', end ='')
        # Prepare Data
        batch_x, _ = sess.run ([x_tensor, y_tensor])
        x_2d_a = rgb_to_y(batch_x[:,:,:,0], batch_x[:,:,:,1], batch_x[:,:,:,2])
        x_b_a = x_2d_a.reshape(batch_size, -1) / 255
    
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer_1, loss_1], feed_dict={x: x_b_a})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

			
    for _ in range(batches_count):
        print (100 * _/batches_count,'%\r', end ='')

        x_2d3c, y_b = sess.run ([x_tensor, y_tensor])
        x_2d = rgb_to_y(x_2d3c[:,:,:,0], x_2d3c[:,:,:,1], x_2d3c[:,:,:,2])
        x_b = x_2d.reshape(batch_size, -1) / 255
        vec_y = np.zeros([batch_size, 3])
        vec_y[range(batch_size), y_b] = 1

        sess.run(train_step, feed_dict={ x: x_b, y: vec_y })

    print ("Testing trained model...")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    sum_accuracy = 0
    for _ in range(tests_count):
        x_test,  y_test = sess.run ([x_test_Tensor, y_test_Tensor])
        x_2d = rgb_to_y(x_2d3c[:,:,:,0], x_2d3c[:,:,:,1], x_2d3c[:,:,:,2])
        x_test = x_2d.reshape(batch_size, -1) / 255  
        resize_test_y = np.zeros ([len(y_test), 3])
        resize_test_y[range(len(y_test)), y_test] = 1   
        sum_accuracy += sess.run(accuracy, feed_dict={x: x_test, y: resize_test_y})
    print("Accuracy: {:f}".format(sum_accuracy / tests_count))
    
    coord.request_stop()
    coord.join(threads)
