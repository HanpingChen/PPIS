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
    def on_epoch_end(self, epoch, logs=None):
        data = self.validation_data
        x = data[0]
        y_true = data[1]
        y_pred = self.model.predict(x)
        pro_pred = y_pred[1]
        y_pred = y_pred[0]
        y = []
        for item in y_true:
            y.extend(x[0] for x in item)
        pred = []
        for item in y_pred:
            pred.extend(np.argmax(item, axis=1))
        l = len(pred)
        print('precision:', precision_score(y, pred, average=None),
              '-', 'recall:', recall_score(y, pred, average=None))
        # 计算回归输出的效果
        y_pro_pred = []
        for item in pro_pred:
            for i in item:
                y_pro_pred.append(int(min(round(i[0]), 2)))
        print(y_pro_pred)
        print('precision:', precision_score(y, y_pro_pred, average=None),
              '-', 'recall:', recall_score(y, y_pro_pred, average=None))


eval_callback = EvalCallback()
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


def Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count):
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    word_emb = Embedding(char_value_dict_len + 1, output_dim=200,
                         input_length=maxlen, name='word_emb')(word_input)

    # bilstm
    bilstm = Bidirectional(LSTM(200, return_sequences=True))(word_emb)
    bilstm_d = Dropout(0.1)(bilstm)

    # cnn
    half_window_size = 2
    padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)
    conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1,
                  padding='valid')(padding_layer)
    conv_d = Dropout(0.1)(conv)
    dense_conv = TimeDistributed(Dense(300))(conv_d)

    # merge
    rnn_cnn_merge = Concatenate(axis=2)([bilstm_d, dense_conv])
    dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)

    # 回归概率输出
    pro_output = TimeDistributed(Dense(1, activation='linear'), name='pro')(rnn_cnn_merge)
    # 获取输出概率
    #pro_output = Lambda(slice_site_pro)(pro_dense)
    # crf
    crf = CRF(class_label_count, sparse_target=True)
    crf_output = crf(dense)
    # # 两个输出相加
    # output = Lambda(add_pro([crf_output, pro_output]))
    # build model
    model = Model(input=word_input, output=[crf_output, pro_output])

    model.compile(loss=[crf.loss_function, keras.losses.MSE], optimizer='adam', metrics=[crf.accuracy])

    # model.summary()

    return model


maxlen, char_value_dict_len, class_label_count = 300, 20, 3
model = Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count)
model.summary()

print(model.input_shape)
print(model.output_shape)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# train
x_train, y_train, x_val, y_val = load_data("config/path_config.json", load_val=True)
y_train_pro = []
for item in y_train:
    temp = []
    for i in range(len(item)):
        if item[i][0] == 2:
            temp.append(5)
        else:
            temp.extend(item[i])
    y_train_pro.extend(temp)
y_train_pro = np.reshape(y_train_pro, [len(y_train), len(y_train[0]), 1])
model.fit(x_train, [y_train, y_train_pro], validation_split=0.1, batch_size=32, epochs=50, verbose=1,callbacks=[eval_callback])