# author：chenhanping
# date 2018/12/13 上午9:42
# copyright ustc sse
import keras
from sklearn.metrics import *
from keras.callbacks import Callback
from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF
from util.data_util import *
from text_process.text_data_generator import *

from keras import backend as K

from keras.utils import plot_model
from keras.utils import np_utils
import math


class EvalCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        data = self.validation_data
        x = data[0]
        y_true = data[1]
        y_pred = self.model.predict(x)
        y = []
        for item in y_true:
            y.append(item[0])
        pred = []
        for item in y_pred:
            if item[0] >= 0.5:
                pred.append(1)
            else:
                pred.append(0)
        print(y)
        print('precision:', precision_score(y, pred, average=None),
              '-', 'recall:', recall_score(y, pred, average=None))


def f1(y_true, y_pred):
    pr = precision(y_true, y_pred)
    re = recall(y_true, y_pred)
    return 2 * ((pr * re) / (pr + re + K.epsilon()))


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    pr = true_positives / (predicted_positives + K.epsilon())
    return pr


eval_callback = EvalCallback()
# 学习率自动调节
lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
# 模型保存回调函数
check_point = keras.callbacks.ModelCheckpoint("model-v6.hdf5", save_best_only=True)
# input:
# maxlen  char_value_dict_len  class_label_count


def slice_site_pro(x):
    """
    返回预测为位点的概率
    :param x:
    :return:
    """
    return x[:, :, 2]


def add_pro(inputs):
    x, y = inputs
    return x + y


def Bilstm_CNN(maxlen, char_value_dict_len, class_label_count):
    # 输入
    inputs = Input(shape=(maxlen,))
    # 嵌入层，将文本转换成词向量表示
    word_embed = Embedding(char_value_dict_len, output_dim=512, input_length=maxlen)
    embedding = word_embed(inputs)
    # cnn
    half_window_size = 3
    padding_layer = ZeroPadding1D(padding=half_window_size)(embedding)
    conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1,
                  padding='valid')(padding_layer)
    conv = BatchNormalization()(conv)
    conv_d = Dropout(0.3)(conv)

    padding_layer = ZeroPadding1D(padding=half_window_size)(conv_d)
    conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1,
                  padding='valid')(padding_layer)
    conv = BatchNormalization()(conv)
    conv_d = Dropout(0.3)(conv)
    dense_conv = Dense(200)(conv_d)
    feature = Concatenate(axis=2)([dense_conv, embedding])
    # BiLSTM
    bilstm = Bidirectional(LSTM(200, return_sequences=False))(feature)
    bilstm_d = Dropout(0.3)(bilstm)

    # many to one
    # feature = Flatten()(dense_conv)
    # total_feature = Concatenate(axis=1)([bilstm_d, feature])
    x = Dense(128, activation='relu')(bilstm_d)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    # build model
    model = Model(input=inputs, output=outputs)
    from keras.optimizers import Adam
    adam = Adam(lr=0.001)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=adam, metrics=['accuracy', precision, recall])

    # model.summary()

    return model


maxlen, char_value_dict_len, class_label_count = 21, 20, 2
model = Bilstm_CNN(maxlen, char_value_dict_len, class_label_count)
model.summary()

print(model.input_shape)
print(model.output_shape)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# # train
x_train, y_train = flow_from_directory("/Users/chenhanping/Downloads/dset4000/text_data/")
x_test, y_test = flow_from_directory("/Users/chenhanping/Downloads/dset72/text_data/")
p = np.random.permutation(range(len(x_train)))
x_train, y_train = x_train[p],y_train[p]
p = np.random.permutation(range(len(x_test)))
x_test, y_test = x_test[p],y_test[p]
x_train = x_train[0: 100000]
y_train = y_train[0: 100000]
print(y_train)
print(np.shape(x_train))
print(np.shape(y_train))
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=64, epochs=200, verbose=1,
          callbacks=[eval_callback, lr_callback, check_point], shuffle=True, class_weight={0: 1, 1: 3})
# model.save('model_v7')