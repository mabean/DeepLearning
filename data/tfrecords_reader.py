import tensorflow as tf
import numpy as np

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320

def read (filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    labels = tf.cast(features['label'], tf.int32)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    resized_images = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)
    return resized_images, labels

def next_batch (images, labels, batch_size):
    x_batch,y_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=1000, num_threads=1, min_after_dequeue=300)
    image_shape = tf.stack([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH * 3])
    x_batch = tf.reshape(x_batch, image_shape)
    return x_batch, y_batch

def next_batch3d(images, labels, batch_size):
    x_batch,y_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=1000, num_threads=1, min_after_dequeue=300)
    return x_batch, y_batch