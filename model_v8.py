# author：chenhanping
# date 2019/1/9 下午2:21
# copyright ustc sse
# 多输入模型，输入为两个蛋白质链p1是需要预测位点的序列，p2是反应的目标序列

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


def embedding_pair(inputs, input_dim, maxlen, out_dim):
    embedding = Embedding(input_dim, out_dim, input_length=maxlen)
    return [embedding(x) for x in inputs]


def conv_bn(inputs, filters=10, kernel_size=3):
    padding_size = int((kernel_size - 1) / 2)
    inputs = ZeroPadding1D(padding=padding_size)(inputs)
    inputs = Conv1D(filters=filters, kernel_size=kernel_size)(inputs)
    x = BatchNormalization()(inputs)
    return x


def deep_conv_bn(inputs, layer_counts, nb_filters, kernel_sizes):
    x = inputs
    for i in range(layer_counts):
        x = conv_bn(x, nb_filters[i], kernel_sizes[i])
    return x


def conv_pair_bn(pair_embedding):
    nb_filters = [64, 64, 64, 32, 32]
    kernel_sizes = [21, 11, 9, 7, 5]
    layer_counts = 5
    xs = []
    for embeb in pair_embedding:
        x = deep_conv_bn(embeb, layer_counts, nb_filters, kernel_sizes)
        xs.append(x)
    return xs


def multi_input_model(maxlen, char_value_dict_len, class_label_count):
    inputs1 = Input(shape=(maxlen, ))
    inputs2 = Input(shape=(maxlen, ))
    # embedding层
    pair_embedding = embedding_pair([inputs1, inputs2], char_value_dict_len, maxlen, 512)
    # 卷积层，提取局部特征
    xs = conv_pair_bn(pair_embedding)
    # BiLSTM，获取序列全局特征
    # 获取需要预测位点的蛋白质
    embed1 = pair_embedding[0]
    # 输入BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(embed1)
    x = Dropout(0.1)(x)
    # merge
    for item in xs:
        x = Concatenate(axis=2)([x, item])

    # crf
    crf = CRF(class_label_count, sparse_target=True)
    crf_output = crf(x)

    # build model
    model = Model(input=[inputs1, inputs2], output=crf_output)
    adam = keras.optimizers.Adam()
    model.compile(loss=crf.loss_function, optimizer=adam, metrics=[crf.accuracy])
    return model


if __name__ == '__main__':
    maxlen, char_value_dict_len, class_label_count = 600, 20, 2
    model = multi_input_model(maxlen, char_value_dict_len, class_label_count)
    model.summary()
    plot_model(model, to_file='model_v8.png', show_shapes=True, show_layer_names=True)
