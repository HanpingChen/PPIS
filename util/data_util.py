# author：chenhanping
# date 2018/12/3 上午10:44
# copyright ustc sse
import json
import os
from keras.preprocessing.text import *
from keras.preprocessing import sequence
import numpy as np


def load_data(config_file_path="../config/path_config.json", load_val=False):
    """
    加载数据
    :param load_val:
    :param config_file_path:
    :return:
    """
    tokenizer = Tokenizer(num_words=20)
    with open(config_file_path, 'r') as f:
        paths = json.load(f)
    label_path = paths['label_path']
    protein_seq_path = paths['protein_seq_path']
    x, y, max_len = read_data(protein_seq_path, label_path)
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
    x_seq = sequence.pad_sequences(x_seq, maxlen=max_len, padding='post')
    y = sequence.pad_sequences(y, maxlen=max_len, padding='post')
    y = np.expand_dims(y, axis=2)
    print(np.shape(x_seq), np.shape(y))
    print(x_seq)
    if load_val:
        val_label_path = paths['dset72_label_path']
        val_seq_path = paths['dset72_protein_seq_path']
        val_x, val_y, _ = read_data(val_seq_path, val_label_path)
        val_texts = list()
        for s in val_x:
            text = ""
            for c in s:
                text += c + " "
            val_texts.append(text)
        val_x_seq = tokenizer.texts_to_sequences(val_texts)
        print(val_x_seq)
        val_x_seq = sequence.pad_sequences(val_x_seq, maxlen=max_len,padding='post')
        val_y = sequence.pad_sequences(val_y, maxlen=max_len,padding='post')
        val_y = np.expand_dims(val_y, axis=2)
        print(np.shape(val_x_seq), np.shape(val_y))
        print(val_x_seq)
        return x_seq, y, val_x_seq, val_y
    return x_seq, y


def read_data(seq_path, label_path):
    max_len = 0
    protein_file_list = os.listdir(seq_path)
    x = []
    y = []
    for file in protein_file_list:
        file_path = os.path.join(seq_path, file)
        filename, _ = os.path.splitext(file)
        f = open(file_path, 'r')
        seq = f.readline()
        if len(seq) > 300:
            continue
        label_file_path = os.path.join(label_path, filename+".txt")
        f = open(label_file_path, 'r')
        label = [int(x)+1 for x in f.readline()]

        if len(seq) > max_len:
            max_len = len(seq)
        x.append(seq)
        y.append(label)
    return x, y, max_len


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
        print(filename)
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


def filter_protein_seq(config_file_path="../config/path_config.json"):
    with open(config_file_path, 'r') as f:
        paths = json.load(f)
    seq_path = paths['protein_seq_path']
    site_path = paths['site_path']
    val_seq_path = paths['val_protein_seq_path']
    seq_file_list = os.listdir(seq_path)
    count = 0
    for file in seq_file_list:
        # if not os.path.exists(os.path.join(site_path, file)):
        #     print(os.path.join(seq_path, file))
        #     os.remove(os.path.join(seq_path, file))
        #     count += 1
        if os.path.exists(os.path.join(val_seq_path, file)):
            count += 1
            print(os.path.join(val_seq_path, file))
            os.remove(os.path.join(seq_path, file))
    print(count)




if __name__ == '__main__':
    create_label()
    # filter_protein_seq()