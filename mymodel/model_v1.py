# inception模型
import numpy
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras_applications.xception import Xception
from keras import backend as K, optimizers
from keras.optimizers import *
import keras
import os


class ModelInception:

    def precision(self, y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc

    def recall(self, y_true, y_pred):
        # 预测
        class_pred = K.argmax(y_pred, axis=-1)
        # 所有的正样本
        class_true = K.argmax(y_true, axis=-1)
        # 预测为正样本的
        accuracy_mask = K.cast(K.equal(class_pred, 1), 'int32')
        # 预测正确的正样本
        class_acc_tensor = K.cast(K.equal(class_true, class_pred), 'int32') * accuracy_mask
        # 所有正样本
        P = K.cast(K.equal(class_true, 1), 'int32')
        recall = K.sum(class_acc_tensor) / K.maximum(K.sum(P), 1)
        return recall

    def f1(self, y_true, y_pred):
        pre = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        pre = K.cast(pre, 'float32')
        r = K.cast(r, 'float32')
        return 2 * (pre * r) / (pre + r)

    def __init__(self):
        self.lr = 0.0001
        self.num_class = 2
        self.weight_path = "model_v1.hdf5"

    def get_model(self):
        model = InceptionV3(weights=None, classes=self.num_class)

        return model

    def train(self, train_data_path, test_data_path='data/ppis/test/',
              epoch=100, model_file_path="model_v1.hdf5", weight_path=None):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(  # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
            train_data_path,  # 训练数据路径，train 文件夹下包含每一类的子文件夹
            target_size=(150, 150),  # 图片大小resize成 150x150
            batch_size=32,
            class_mode='categorical')  # 使用二分类，返回2-D 的二值标签
        test_datagen = ImageDataGenerator(
            rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(  # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
            test_data_path,  # 训练数据路径，train 文件夹下包含每一类的子文件夹
            target_size=(150, 150),  # 图片大小resize成 150x150
            batch_size=32,
            class_mode='categorical')  # 使用二分类，返回2-D 的二值标签
        model = self.get_model()
        model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy',
                      metrics=['accuracy', self.precision, self.recall, self.f1])
        if weight_path is None:
            if os.path.exists(self.weight_path):
                model.load_weights(self.weight_path)
        else:
            model.load_weights(weight_path)
        model_checkpoint = ModelCheckpoint(model_file_path, monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(train_generator,
                            callbacks=[model_checkpoint, self.show_tb],
                            epochs=epoch,
                            class_weight='auto',
                            validation_data=test_generator,
                            shuffle=True,
                            steps_per_epoch=100
                            )

    def show_tb(self, log_path='E:\PPIS\log'):
        """
        tensorboard
        :return:
        """
        t_callback = keras.callbacks.TensorBoard(log_dir=log_path,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)
        return t_callback
