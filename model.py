# author：chenhanping
# date 2018/11/16 上午11:36
# copyright ustc sse
import numpy
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K, optimizers
import keras

from image_process.load_data import load_data

train_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(  # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
    'data/ppis/train/',  # 训练数据路径，train 文件夹下包含每一类的子文件夹
    target_size=(150, 150),  # 图片大小resize成 150x150
    batch_size=32,
    class_mode='categorical')  # 使用二分类，返回2-D 的二值标签
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(  # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
    'data/ppis/test/',  # 训练数据路径，train 文件夹下包含每一类的子文件夹
    target_size=(150, 150),  # 图片大小resize成 150x150
    batch_size=32,
    class_mode='categorical')  # 使用二分类，返回2-D 的二值标签
# print(train_generator.class_indices)
model = InceptionV3(weights=None, classes=2)

from sklearn.metrics import precision_score, recall_score, f1_score


class Metrics(Callback):
    import numpy as np
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        batch_result = self.model.predict(self.validation_data[0])
        val_predict = numpy.argmax(batch_result, axis=1)
        val_targ = self.validation_data[1][:,1]
        print("pred", val_predict)
        print("y_true", val_targ)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("recall", _val_recall)
        return
metrics = Metrics()


tbCallBack = keras.callbacks.TensorBoard(log_dir='log',
histogram_freq=1,
write_graph=True,
write_images=True)
# #
from keras.optimizers import Adam
# model.load_weights("model_v1.hdf5")
sgd = optimizers.SGD(lr=0.001, decay=0.09, momentum=0.8, nesterov=True)
model_checkpoint = ModelCheckpoint('model_v1.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# X, Y = load_data(10000, "tfrecord/train.tfrecord")
# X_val, Y_val = load_data(2000, "tfrecord/test.tfrecord")
model.fit_generator(train_generator,
                    callbacks=[model_checkpoint, metrics],
                    epochs=10,
                    class_weight={1: 3, 0: 1},
                    validation_data=test_generator,
                    shuffle=True,

)
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=200,
#         epochs=20,
#         callbacks=[model_checkpoint])
#
# test_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
#
# test_generator = test_datagen.flow_from_directory(  # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
#     'data/test/',  # 训练数据路径，train 文件夹下包含每一类的子文件夹
#     target_size=(150, 150),  # 图片大小resize成 150x150
#     batch_size=32,
#     class_mode='categorical')  # 使用二分类，返回1-D 的二值标签


# result = model.predict_generator(test_generator)
# score = model.evaluate_generator(train_generator,steps=128)
# print(score)
# print(result)