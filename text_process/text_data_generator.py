# author：chenhanping
# date 2018/11/28 下午2:27
# copyright ustc sse
import os
import json

from keras.preprocessing import sequence
from keras.preprocessing.text import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def get_tf_idf():
    data, label = text_generate(is_process=False)
    vectorizer = CountVectorizer(min_df=1e-5)  # drop df < 1e-5,去低频词
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data))
    words = vectorizer.get_feature_names()
    return tfidf, label


def get_data_path(path, protein_seq_name="protein_seq",
                  index_dict_name="index_dict",
                  site_set_name="site_set")->list:
    """
    获取所需的数据所在的路径
    :param site_set_name: 位点路径
    :param index_dict_name: res id和序列index对应字典
    :param protein_seq_name: 蛋白质序列
    :param path:
    :return: 所有数据的路径
    """
    # 蛋白质序列的路径
    seq_path = os.path.join(path, protein_seq_name)
    # res id和序列index对应字典的路径
    index_dict_path = os.path.join(path, index_dict_name)
    # 存储位点文件的路径
    site_path = os.path.join(path, site_set_name)
    paths = [seq_path, index_dict_path, site_path]
    return paths


def get_protein_seq(path):
    """
    读取蛋白序列
    :param path:
    :return:
    """
    f = open(path)
    seq = f.readline()
    return seq


def get_relevant_aa_seq(seq, index, window_size=21):
    """
    获取一个氨基酸相邻的序列，用于做位点预测
    根据滑动窗口法，可以选择与这个氨基酸左右各10个氨基酸
    :param window_size: 滑动窗口的大小，截取index-(window_size - 1) / 2 到index+(window_size - 1) / 2 之间的序列
    :param index: 在肽链序列上的index
    :param seq: 完整的肽链
    :return: 序列
    """
    # 开始截取的位置为index - (window_size - 1)或者是0，当index - (window_size - 1)<0是就从0开始截取
    start = int(max(index - (window_size - 1) / 2, 0))
    # 截取的结尾是1 + start + (window_size - 1) / 2，当1 + start + (window_size - 1) / 2比seq长度大时，选择最后一位
    end = int(min(1 + index + (window_size - 1) / 2, len(seq)))
    sub_seq = seq[start:end]
    return sub_seq


def get_protein_chain_name(file):
    """
    分割出蛋白质名字和肽链的名字
    :param file:
    :return:
    """
    filename, _ = os.path.splitext(file)
    protein_name, chain, = filename.split("_")
    return protein_name, chain


def get_index_dict(path):
    with open(path, "r") as f:
        return json.load(f)


def get_site_set(path):
    f = open(path)
    site = f.readline()
    site_arr = [int(x) for x in site.split(",")]
    return set(site_arr)


def get_chain_data(index_dict, site_set, seq, window_size=21):
    """
    生成一个肽链的序列数据，一个氨基酸生成一个序列，相当于一句话，以及label
    :param index_dict:
    :param site_set:
    :param seq:
    :return:
    """
    label = []
    data = []
    for item in index_dict.items():
        res_id = int(item[0])
        index = item[1]
        data.append(get_relevant_aa_seq(seq, index, window_size))
        if res_id in site_set:
            # 是结合位点
            label.append(1)
        else:
            label.append(0)
    return data, label


def get_text_data(protein_seq_path,
                  index_dict_path,
                  site_set_path,
                  window_size=21):
    # 遍历序列文件
    seq_file_list = os.listdir(protein_seq_path)
    # 序列集合，每一个位点一条序列
    seqs = list()
    # 类别，每一个位点的类别
    labels = list()
    for file in seq_file_list:
        file_path = os.path.join(protein_seq_path, file)
        # 分割出文件名
        filename, _ = os.path.splitext(file)
        # 获取index_dict文件
        index_dict_file_path = os.path.join(index_dict_path, filename+".json")
        index_dict = get_index_dict(index_dict_file_path)
        # 获取site set文件
        site_set_file_path = os.path.join(site_set_path, filename+".txt")
        # 获取位点set
        site_set = get_site_set(site_set_file_path)
        seq = get_protein_seq(file_path)
        # 获取肽链数据
        data, label = get_chain_data(index_dict, site_set, seq, window_size)
        for d in data:
            seqs.append(d)
        for l in label:
            labels.append(l)
    print(seqs)
    print(labels)
    print(len(seqs), len(labels))
    return seqs, labels


def text_generate(path="/Users/chenhanping/Downloads/rnn_data/", save_path=None,
                  protein_seq_name="protein_seq",
                  index_dict_name="index_dict",
                  site_set_name="site_set",
                  window_size=21, is_process=True):
    """
    生成文本数据，
    :param window_size: 滑动窗口的大小
    :param site_set_name: 位点路径
    :param index_dict_name: res id和序列index对应字典
    :param protein_seq_name: 蛋白质序列
    :param path: 数据存储地址，包括蛋白质序列，index_dict, 和site set 三个文件夹
    :param save_path: 生成的文本存储地址
    :return:
    """
    paths = get_data_path(path, protein_seq_name, index_dict_name, site_set_name)
    # 判断数据是否齐全
    for p in paths:
        if not os.path.exists(p):
            print(p, "不存在")
            return
    # 生成代表每一个位点的序列以及他的label
    seq_data, labels = get_text_data(paths[0], paths[1], paths[2], window_size=window_size)
    if save_path is not None:
        print("saving")
        # 将文本保存
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 负样本保存
        n_path = os.path.join(save_path, '0')
        os.makedirs(n_path, exist_ok=True)
        # 正样本
        p_path = os.path.join(save_path, '1')
        os.makedirs(p_path, exist_ok=True)
        save_file = ""
        for i in range(len(seq_data)):
            if labels[i] == 0:
                save_file = os.path.join(n_path, str(i)+".txt")

            else:
                save_file = os.path.join(p_path, str(i)+".txt")
            f = open(save_file, 'w')
            f.write(seq_data[i])
        print('complete')
    if is_process:
        return process(seq_data, labels)
    return seq_data, labels


def process(x, y):
    tokenizer = Tokenizer(num_words=30)  # 建立一个20个单词的字典
    texts = list()
    for s in x:
        text = ""
        for c in s:
            text += c + " "
        texts.append(text)
    tokenizer.fit_on_texts(texts)
    # 对每个字符串转换为数字列表，使用每个词的编号进行编号
    x_train_seq = tokenizer.texts_to_sequences(texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=7)
    y_train = np.array(y)
    return x_train, y_train


def flow_from_directory(directory):
    dict = os.listdir(directory)
    data = []
    labels = []
    i = 0
    for d in dict:
        ld = os.path.join(directory, d)
        if not os.path.isdir(ld):
            continue
        file_list = os.listdir(ld)
        count = 0
        for file in file_list:
            count += 1
            if count > 6000:
                break
            f = open(os.path.join(ld, file))
            seq = f.readline()
            data.append(seq)
            labels.append(i)
        i += 1

    return process(data, labels)


if __name__ == '__main__':
    path = "/Users/chenhanping/Downloads/rnn_data"
    save_path = "/Users/chenhanping/Downloads/rnn_data/text_data/"
    # text_generate(path, save_path=save_path)
    # x, y = flow_from_directory(save_path)
    get_tf_idf()


