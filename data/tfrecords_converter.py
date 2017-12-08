import tensorflow as tf
import csv
import os
from PIL import Image
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _write_data(image_path, label, writer):
    image = np.array(Image.open(image_path))
    rows = image.shape[0]
    cols = image.shape[1]

    image_raw = image.tostring()
    print("Write ", image_path)
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())

def _read_data_from_folder(directory, folder):
    os.chdir('..')
    train_writer = tf.python_io.TFRecordWriter(os.path.join(os.getcwd(), 'dataset_train.tfrecords'))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(os.getcwd(), 'dataset_test.tfrecords'))
    os.chdir(folder)
    for file in os.listdir(directory):
        entry = os.path.join(directory, file)
        if (os.path.isdir(entry)):
            continue
        photo_dir = os.path.splitext(entry)[0]
        with open(entry) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for idx, row in enumerate(csv_reader):
                if (len(row) != 5):
                    print(entry, idx)
                    raise Exception()
                _write_data(image_path=os.path.join(directory, row[3]), label=row[4],
                            writer=test_writer if idx % 5 == 0 else train_writer)
    train_writer.close()
    test_writer.close()
                    
def convert(folder):
    os.chdir(folder)
    _read_data_from_folder('data', folder)
    os.chdir('..')