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


from util.call_back import *


def read_data(path, max_len):
    lines = [x.rstrip("\n") for x in open(path).readlines()]
    print(lines)
    p1 = []
    p2 = []
    label = []
    for i in range(0, len(lines), 4):
        p1_seq = lines[i]
        p1_label = [str(int(x)+1) for x in lines[i+1]]
        p2_seq = lines[i+2]
        p2_label = [str(int(x)+1) for x in lines[i+3]]
        p1.append(p1_seq)
        p2.append(p2_seq)
        label.append(p1_label)
        p1.append(p2_seq)
        p2.append(p1_seq)
        label.append(p2_label)
    texts = list()
    for s in p1:
        text = ""
        for c in s:
            text += c + " "
        texts.append(text)
    tokenizer = Tokenizer(num_words=20)
    tokenizer.fit_on_texts(texts)
    print(tokenizer.word_counts)
    print(tokenizer.word_index)
    x1_seq = tokenizer.texts_to_sequences(texts)
    x1_seq = sequence.pad_sequences(x1_seq, maxlen=max_len, padding='post')
    texts = list()
    for s in p2:
        text = ""
        for c in s:
            text += c + " "
        texts.append(text)
    x2_seq = tokenizer.texts_to_sequences(texts)
    x2_seq = sequence.pad_sequences(x2_seq, maxlen=max_len, padding='post')
    y = sequence.pad_sequences(label, maxlen=max_len, padding='post')
    y = np.expand_dims(y, axis=2)
    return x1_seq, x2_seq, y


if __name__ == '__main__':

    p1, p2, label = read_data("/Users/chenhanping/Downloads/dataset/train14000.txt", 600)

    maxlen, char_value_dict_len, class_label_count = 600, 20, 2
    model = multi_input_model(maxlen, char_value_dict_len, class_label_count)
    model.summary()
    plot_model(model, to_file='model_v8.png', show_shapes=True, show_layer_names=True)
    lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    # 模型保存回调函数
    check_point = keras.callbacks.ModelCheckpoint("model-v8.hdf5", save_best_only=True)
    eval = EvalCallback()
    model.fit([p1, p2], label, validation_split=0.05, callbacks=[eval])