# author：chenhanping
# date 2017/11/30 上午11:57
# copyright ustc sse
# 测试写tfrecord
#DATA_DIR = '../data/train/'
#TF_RECORD_DIR = '../tfrecord/'
DATA_DIR = "E:\\PPIS\\data\\train\\"
TF_RECORD_DIR = "E:\\PPIS\\tfrecord\\"
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import random
"""
遍历文件夹下的所有文件夹，每一个文件夹的文件夹的名字代表的一个类别
将类别记录到label，并用label_dict 存储label的对应关系
"""
label_dict = {}
index = 0
tfrecord_file_name = TF_RECORD_DIR+"train.tfrecord"
# 创建一个tfrecord write对象
if not os.path.exists(TF_RECORD_DIR):
    os.makedirs(TF_RECORD_DIR)
writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
for label_name in os.listdir(DATA_DIR):
    image_dir = os.path.join(DATA_DIR, label_name)
    if os.path.isdir(image_dir):
        # 是文件夹，文件夹中存放着图片，则获取文件夹的名字
        print("正在写" + label_name)
        # 用数字代表label
        label_dict[label_name] = index
        # 读取这个文件夹
        image_dir_list = os.listdir(image_dir)
        print(len(image_dir_list))
        image_list = random.sample(range(len(image_dir_list)), len(image_dir_list))

        for i in image_list:
            # 读取出numpy array
            image_name = image_dir_list[i]
            # 获取图像文件的路径
            image_file_path = os.path.join(image_dir, image_name)
            image_origin = cv.imread(image_file_path)

            # 将图像矩阵转换成字符串
            try:
                image = cv.resize(image_origin, (150, 150))
                image_raw = image.tobytes()
                # 创建一个样本对象
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                }
                ))
                # 将创建的样本写入tfrecord
                writer.write(example.SerializeToString())
            except:
                print(image_file_path)
        index += 1
writer.close()

