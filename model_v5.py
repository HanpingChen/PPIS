# author：chenhanping
# date 2018/12/7 下午2:38
# copyright ustc sse
# author：chenhanping
# date 2018/12/7 上午11:39
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


class EvalCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        data = self.validation_data
        x = data[0]
        y_true = data[1]
        y_pred = self.model.predict(x)
        y = []
        for item in y_true:
            for val in item:
                y.append(max(np.argmax(val, axis=0)-1, 0))
        pred = []
        site_pro = []
        for item in y_pred:
            for val in item:
                site_pro.append(val[2])
        sorted_site_pro = sorted(site_pro, reverse=True)
        #阈值
        d_val = min(sorted_site_pro[int(len(sorted_site_pro)*0.12)], 0.1)
        for val in site_pro:
            if val >= d_val:
                pred.append(1)
            else:
                pred.append(0)
        l = len(pred)
        pred = pred[l - len(y):]
        print('precision:', precision_score(y, pred, average=None),
              '-', 'recall:', recall_score(y, pred, average=None),
              '-', 'acc', accuracy_score(y, pred))


eval_callback = EvalCallback()

import tensorflow as tf


# input:
# maxlen  char_value_dict_len  class_label_count
def Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count, weight_metric):
    word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
    word_emb1 = Embedding(char_value_dict_len + 1, output_dim=100, weights=[weight_metric],trainable=False,
                         input_length=maxlen, name='word_emb1')(word_input)

    word_emb2 = Embedding(char_value_dict_len + 1, output_dim=100,
                         input_length=maxlen, name='word_emb2')(word_input)

    # 两个emb融合
    word_emb = Concatenate(axis=2, name='word_emb')([word_emb1, word_emb2])
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

    output1 = TimeDistributed(Dense(3, activation='softmax'))(rnn_cnn_merge)

    # build model
    model = Model(input=word_input, output=output1)
    from keras.losses import binary_crossentropy, categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # model.summary()

    return model


maxlen, char_value_dict_len, class_label_count = 300, 20, 3
x_train, y_train, x_val, y_val, tokenizer = load_data("config/path_config.json", load_val=True, get_token=True)
from text_process.word2vector_train import *

model = Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count,
                       get_weight_metric(tokenizer,'text_process/veb.txt'))
model.summary()

print(model.input_shape)
print(model.output_shape)

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# train
weight = get_weight_metric(tokenizer, 'text_process/veb.txt')
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
model.load_weights('model-v5.hdf5')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=32, epochs=200, verbose=1,
          callbacks=[eval_callback])
model.save("model-v5.hdf5")