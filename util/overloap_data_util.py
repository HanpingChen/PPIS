# author：chenhanping
# date 2018/12/25 下午3:18
# copyright ustc sse
# 对少量数据进行overloap操作，将一个序列分解为多个序列
import os
from keras.preprocessing import sequence
from keras_preprocessing.text import Tokenizer
import numpy as np

max_len = 100


def get_tokenizer(x):
    tokenizer = Tokenizer(num_words=20)  # 建立一个20个单词的字典
    tokenizer.fit_on_texts(x)
    print(tokenizer.word_index)
    return tokenizer


def read_txt(seq_path, label_path):
    file_list = os.listdir(seq_path)
    x = list()
    y = list()
    for file in file_list:
        seq = open(os.path.join(seq_path, file)).readline()
        text = ""
        for s in seq:
            text += s + " "
        label = [int(x) for x in open(os.path.join(label_path,file)).readline()]
        x.append(text)
        y.append(label)
    return x, y


def split_from_directory(data_directory):
    seq_path = os.path.join(data_directory, "protein_seq")
    label_path = os.path.join(data_directory, "label")
    seqs, labels = read_txt(seq_path, label_path)
    t = get_tokenizer(seqs)
    #seqs = t.texts_to_sequences(seqs)
    x = list()
    y = list()
    for (seq, label) in zip(seqs, labels):

        if len(seq) < max_len:
            # 直接返回序列
            x.append(seq)
            y.append(label)
            continue
            #yield seq, label
        # 分割，
        for i in range(len(seq) - max_len):
            sub_seq = seq[i: i + max_len + 1]
            sub_label = label[i: i + max_len + 1]
            x.append(sub_seq)
            y.append(sub_label)
            #yield sub_seq, sub_label
    x = t.texts_to_sequences(x)
    x = sequence.pad_sequences(x, maxlen=max_len, padding='post')
    y = sequence.pad_sequences(y, maxlen=max_len, padding='post')
    y = np.expand_dims(y, axis=2)
    return x, y


def split_single_chain(seq, label, t):
    print(seq)


if __name__ == '__main__':
    count = 0
    split_from_directory("/Users/chenhanping/Downloads/rnn_data")
    # for x, y in split_from_directory("/Users/chenhanping/Downloads/rnn_data"):
    #     count += 1
    #     print(x, y)
    # print(count)