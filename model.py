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
    '/Users/chenhanping/data/ppis/train/',  # 训练数据路径，train 文件夹下包含每一类的子文件夹
    target_size=(150, 150),  # 图片大小resize成 150x150
    batch_size=32,
    class_mode='categorical')  # 使用二分类，返回2-D 的二值标签
test_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(  # 以文件夹路径为参数,自动生成经过数据提升/归一化后的数据和标签
    '/Users/chenhanping/data/ppis/test/',  # 训练数据路径，train 文件夹下包含每一类的子文件夹
    target_size=(150, 150),  # 图片大小resize成 150x150
    batch_size=32,
    class_mode='categorical')  # 使用二分类，返回2-D 的二值标签
# print(train_generator.class_indices)
model = InceptionV3(weights=None, classes=2)

from sklearn.metrics import precision_score, recall_score, f1_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='macro')
        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("recall", _val_recall)
        return


metrics = Metrics()


def precision(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc


def recall(y_true, y_pred):

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


def f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    pre = K.cast(pre, 'float32')
    r = K.cast(r, 'float32')
    return 2*(pre * r) / (pre + r)


tbCallBack = keras.callbacks.TensorBoard(log_dir='log',
histogram_freq=1,
write_graph=True,
write_images=True)
# #
from keras.optimizers import Adam
# model.load_weights("model_v1.hdf5")
sgd = optimizers.SGD(lr=0.01, decay=0.09, momentum=0.8, nesterov=True)
model_checkpoint = ModelCheckpoint('model_v1.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1])
# X, Y = load_data(10000, "tfrecord/train.tfrecord")
# X_val, Y_val = load_data(2000, "tfrecord/test.tfrecord")
num_non = 30699
num_site = 5505
# t 在0到1之间，t越大，则约优化recall，t越小，越优化precision
t = 0.48
model.fit_generator(train_generator,
                    callbacks=[model_checkpoint,metrics],
                    epochs=20,
                    class_weight={1: (num_non / num_site)*t, 0: 1},
                    validation_data=test_generator,
                    shuffle=True,

)