# author：chenhanping
# date 2018/12/3 上午10:44
# copyright ustc sse
import json
import os
from keras.preprocessing.text import *
from keras.preprocessing import sequence
import numpy as np


def load_data(config_file_path="../config/path_config.json"):
    """
    加载数据
    :param config_file_path:
    :return:
    """
    tokenizer = Tokenizer(num_words=20)
    with open(config_file_path, 'r') as f:
        paths = json.load(f)
    label_path = paths['label_path']
    protein_seq_path = paths['protein_seq_path']
    protein_file_list = os.listdir(protein_seq_path)
    x = []
    y = []
    max_len = 0
    for file in protein_file_list:
        file_path = os.path.join(protein_seq_path, file)
        filename, _ = os.path.splitext(file)
        f = open(file_path, 'r')
        seq = f.readline()
        if len(seq) > 300:
            continue
        label_file_path = os.path.join(label_path, filename+".txt")
        f = open(label_file_path, 'r')
        label = [int(x) + 1 for x in f.readline()]

        if len(seq) > max_len:
            max_len = len(seq)
        x.append(seq)
        y.append(label)
    texts = list()
    for s in x:
        text = ""
        for c in s:
            text += c + " "
        texts.append(text)
    tokenizer.fit_on_texts(texts)
    print(tokenizer.word_counts)
    print(tokenizer.word_index)
    x_seq = tokenizer.texts_to_sequences(texts)
    print(x_seq)
    x_seq = sequence.pad_sequences(x_seq, maxlen=max_len)
    y = sequence.pad_sequences(y, maxlen=max_len)
    y = np.expand_dims(y, axis=2)
    print(np.shape(x_seq), np.shape(y))
    print(x_seq)
    return x_seq, y


def create_label(config_file_path="../config/path_config.json"):
    with open(config_file_path, 'r') as f:
        paths = json.load(f)
    label_path = paths['label_path']
    protein_seq_path = paths['protein_seq_path']
    index_dict_path = paths['index_dict_path']
    site_set_path = paths['site_path']
    protein_file_list = os.listdir(protein_seq_path)

    for file in protein_file_list:
        label = ""
        filename, _ = os.path.splitext(file)
        index_dict_file = os.path.join(index_dict_path, filename+".json")
        site_set_file = os.path.join(site_set_path, filename + ".txt")
        with open(site_set_file, 'r') as f:
            site_arr = [int(x) for x in f.readline().split(',')]
        site_set = set(site_arr)
        with open(index_dict_file, 'r') as f:
            index_dict = json.load(f)
        for res_id, _ in index_dict.items():
            if int(res_id) in site_set:
                label += '1'
            else:
                label += '0'
        label_file_path = os.path.join(label_path, filename+".txt")
        print(filename, label, len(label))
        # 保存
        f = open(label_file_path, 'w')
        f.write(label)
        break


def process_label(config_file_path="../config/path_config.json"):
    with open(config_file_path, 'r') as f:
        paths = json.load(f)
    label_path = paths['label_path']
    label4_path = paths['label4_path']
    files = os.listdir(label_path)

    for file in files:
        label_file_path = os.path.join(label_path, file)
        f = open(label_file_path, 'r')
        y = [int(x) for x in f.readline()]

        label = [0] * len(y)
        print(len(label))
        if y[0] == 1:
            label[0] = 2
        for i in range(1, len(y)):
            if y[i] == 1:
                if y[i-1] == 0:
                    # 是开始位点
                    label[i] = 2
                else:
                    # 非起始位点
                    label[i] = 3
            if y[i] == 0:
                if y[i-1] == 0:
                    # 非起始
                    label[i] = 1

        label4_file_path = os.path.join(label4_path, file)
        f = open(label4_file_path, 'w')
        label4_str = ''.join(str(x) for x in label)
        f.write(label4_str)
        print(file, label, len(label))


if __name__ == '__main__':
    process_label()