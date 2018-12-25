# author：chenhanping
# date 2018/12/6 下午5:41
# copyright ustc sse
import gc
import numpy as np
# np.random.seed(1111)

import keras
from sklearn.metrics import *
from keras.callbacks import Callback
from keras.layers import *
from keras.models import *
from keras_contrib.layers import CRF
from util.data_util import *

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
            y.extend(x[0] for x in item)
        pred = []
        for item in y_pred:
            for i in item:
                if i[0] >= 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
        l = len(pred)
        print('precision:', precision_score(y, pred, average=None),
              '-', 'recall:', recall_score(y, pred, average=None))



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


import tensorflow as tf
import keras.backend as K

def weighted_loss(y_true, y_pred):

    return K.mean(K.tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 8))



def Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count):
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    word_emb = Embedding(char_value_dict_len + 1, output_dim=200,
                         input_length=maxlen, name='word_emb')(word_input)

    # bilstm
    bilstm = Bidirectional(LSTM(32, return_sequences=True))(word_emb)
    bilstm_d = Dropout(0.1)(bilstm)

    # cnn
    half_window_size = 3
    padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)
    conv = Conv1D(nb_filter=20, filter_length=2 * half_window_size + 1,
                  padding='valid')(padding_layer)
    conv_d = Dropout(0.1)(conv)
    dense_conv = Dense(32)(conv_d)

    # merge
    rnn_cnn_merge = Concatenate(axis=2)([bilstm_d, dense_conv])

    output = Dense(1, activation='sigmoid')(rnn_cnn_merge)
    # 获取输出概率
    #pro_output = Lambda(slice_site_pro)(pro_dense)
    # # 两个输出相加
    # output = Lambda(add_pro([crf_output, pro_output]))
    # build model
    model = Model(input=word_input, output=output)
    # keras.losses.binary_crossentropy()
    model.compile(loss=weighted_loss, optimizer='adam', metrics=['accuracy'])

    # model.summary()

    return model


maxlen, char_value_dict_len, class_label_count = 100, 20, 2
model = Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count)
model.summary()

print(model.input_shape)
print(model.output_shape)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# train
from util.overloap_data_util import *
# x_train, y_train, x_val, y_val = load_data("config/path_config.json", load_val=True)
x_train, y_train = split_from_directory("/Users/chenhanping/Downloads/rnn_data")
print(np.shape(x_train))
print(np.shape(y_train))
x_val, y_val = split_from_directory("/Users/chenhanping/Downloads/dset72")
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=200, verbose=1, callbacks=[eval_callback])
model.save('model-v6.hdf5')