# author：chenhanping
# date 2018/11/28 下午1:53
# copyright ustc sse
from keras.layers import *
from keras.models import *

from text_process.text_data_generator import *


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
    # 所有的实际值
    class_true = K.argmax(y_true, axis=-1)
    # 预测为正样本的
    accuracy_mask = K.cast(K.equal(class_pred, 1), 'int32')
    # 预测正确的正样本
    class_acc_tensor = K.cast(K.equal(class_true, class_pred), 'int32') * accuracy_mask
    # 所有正样本
    P = K.cast(K.equal(class_true, 1), 'int32')
    recall = K.sum(class_acc_tensor) / K.maximum(K.sum(P), 1)
    return recall


def BRNN(max_features=30, embed_size=20, maxlen=21):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True), merge_mode='concat'))
    model.add(SimpleRNN(8))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', metrics=['accuracy', recall, precision], loss='categorical_crossentropy')

    return model


def TextCNN(max_len=21, max_features=30, embed_size=30):
    aa_seq = Input(shape=[max_len], name="input_aa_seq")
    emb_seq = Embedding(max_features, embed_size)(aa_seq)

    # 卷积层
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for f in filter_sizes:
        l_conv = Conv1D(filters=30, kernel_size=f, activation='relu')(emb_seq)
        l_pool = MaxPool1D(max_len-f+1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)
    output = Dense(2, activation='softmax')(output)
    model = Model([aa_seq], output)
    model.compile(optimizer='adam', metrics=['accuracy',recall,precision], loss='binary_crossentropy')
    return model


if __name__ == '__main__':
    import numpy as np
    from keras.utils import *
    winndow_size = 9
    model = TextCNN(max_len=winndow_size)
    # x, y = flow_from_directory("/Users/chenhanping/Downloads/rnn_data/text_data")
    x, y = text_generate(path="data/ppis/rnn_data/rnn_data", window_size=winndow_size)
    print(np.shape(x), np.shape(y))
    y = to_categorical(y)
    # print(np.count_nonzero(y))
    print(y)
    model.fit(x, y,
              shuffle=True,
              epochs=1000,
              validation_split=0.1,
              class_weight={1: 3, 0: 0.9},
              batch_size=32)
    score = model.evaluate(x, y, batch_size=32)
    print(score)
