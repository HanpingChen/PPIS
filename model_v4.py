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
            y.extend(np.argmax(item, axis=1))
        pred = []
        for item in y_pred:
            for val in item:
                if val[2] > 0.2:
                    pred.append(2)
                else:
                    pred.append(np.argmax(val, axis=0))

        l = len(pred)
        pred = pred[l - len(y):]
        print('precision:', precision_score(y, pred, average=None),
              '-', 'recall:', recall_score(y, pred, average=None),
              '-', 'acc', accuracy_score(y, pred))


eval_callback = EvalCallback()

def func(last_output):

    pass
# input:
# maxlen  char_value_dict_len  class_label_count
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

    output1 = TimeDistributed(Dense(3, activation='softmax'))(rnn_cnn_merge)


    # build model
    model = Model(input=word_input, output=output1)
    from keras.losses import binary_crossentropy,categorical_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # model.summary()

    return model


maxlen, char_value_dict_len, class_label_count = 300, 20, 3
model = Bilstm_CNN_Crf(maxlen, char_value_dict_len, class_label_count)
model.summary()

print(model.input_shape)
print(model.output_shape)

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# train
x_train, y_train, x_val, y_val = load_data("config/path_config.json", load_val=True)

y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
#model.load_weights('model-v4.hdf5')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          batch_size=32, epochs=120, verbose=1, shuffle=True,
          callbacks=[eval_callback])
model.save("model-v4.hdf5")