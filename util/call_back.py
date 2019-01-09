# author：chenhanping
# date 2019/1/9 下午4:57
# copyright ustc sse
# 学习率自动调节
import keras
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score
import numpy as np


class EvalCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        data = self.validation_data
        x = data[0]
        y_true = data[1]
        y_pred = self.model.predict(x)
        y = []
        for item in y_true:
            y.extend(x[0] for x in item)
        pred = []
        for item in y_pred:
            pred.extend(np.argmax(item, axis=1))
        l = len(pred)
        pred = pred[l - len(y):]
        print(pred)
        print('precision:', precision_score(y, pred, average=None),
              '-', 'recall:', recall_score(y, pred, average=None))
