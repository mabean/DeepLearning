import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])

hiddenLayer = tf.get_variable('hiddenLayer', [784, 100], initializer=tf.random_normal_initializer())
hiddenBiases = tf.get_variable('hiddenBiases', [1,], initializer=tf.constant_initializer(0.0))

topLayer = tf.get_variable('topLayer', [100, 10], initializer=tf.random_normal_initializer())
topLayerBiases = tf.get_variable('topLayerBiases', [1,], initializer=tf.constant_initializer(0.0))

hiddenLayerOutput = tf.nn.tanh(tf.matmul(X, hiddenLayer) + hiddenBiases)

output = tf.nn.softmax(tf.matmul(hiddenLayerOutput, topLayer) + topLayerBiases)

y = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean((-tf.reduce_sum(y * tf.log(output), reduction_indices=[1])))
train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.2).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(100000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, y: mnist.test.labels}))
