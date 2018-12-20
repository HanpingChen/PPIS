# coding=utf-8

# -*- coding: UTF-8 -*-
# author：chenhanping
# date 2017/11/30 下午5:55
# copyright ustc sse
import tensorflow as tf
import numpy as np
MIN_AFTER_QUEUE = 200


def load_data(batch_size, record_file_path):
    """

    :param batch_size:
    :param record_file_path:
    :return:
    """
    # 获取读取tfrecord的对象
    reader = tf.TFRecordReader()
    file_path = tf.train.string_input_producer([record_file_path])
    _, serialized_example = reader.read(file_path)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [150, 150, 3])
    label = tf.cast(features['label'], tf.int64)

    images, labels = tf.train.shuffle_batch([image, label],
                                                        capacity=10000,min_after_dequeue=MIN_AFTER_QUEUE,
                                                        batch_size=batch_size, num_threads=2)

    labels = tf.one_hot(labels, 2, 1, 0)
    labels = tf.cast(labels, dtype=tf.int32)
    labels = tf.reshape(labels, [batch_size, 2])
    init = tf.local_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    images, labels = sess.run([images, labels])
    coord.request_stop()
    coord.join(threads)
    return images, labels


if __name__ == '__main__':
    load_data(64)