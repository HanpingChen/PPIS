# author：chenhanping
# date 2018/12/3 上午10:40
# copyright ustc sse
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from util.data_util import *
import pickle
import keras.backend as K

EMBED_DIM = 200
BiRNN_UNITS = 200


def precision(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    accuracy_mask = K.cast(K.equal(class_id_preds, 1), K.floatx())
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), K.floatx()) * accuracy_mask
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

def create_model(train=True, config_path="config/path_config.json"):
    if train:
        train_x, train_y = load_data(config_path)
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)
    model = Sequential()
    model.add(Embedding(20, EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(2, sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    #model.load_weights('model_v2.hdf5')
    model.fit(train_x, train_y, epochs=100, validation_split=0.1, shuffle=True, batch_size=32)
    test = train_x[0:1]
    print(model.predict(test))
    model.save('model_v2.hdf5')


if __name__ == '__main__':
    create_model()