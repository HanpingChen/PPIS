# author：chenhanping
# date 2018/12/7 下午4:53
# copyright ustc sse
from gensim.models import *
import os
from Bio import SeqIO


def load_seq(path="/Users/chenhanping/data/ppis/chain_seq"):
    seq_list = os.listdir(path)
    sentences = []
    for file in seq_list:
        sentence = []
        seq = open(os.path.join(path, file)).readline()
        sentences.append(seq)
    return sentences

def load_model():
    model = Word2Vec.load('veb.txt')
    model.most_similar('A')

import numpy as np
def get_weight_metric(tokenizer, path='veb.txt'):
    model = Word2Vec.load(path)
    word_index = tokenizer.word_index
    weight = np.zeros(shape=(21, 100))
    for c, index in word_index.items():
        if c == 'x':
            continue
        weight[index] = model[c.upper()]

    print(weight)
    return weight

def train():
    sentences = load_seq()
    model = Word2Vec(sentences)
    model.save("veb.txt")
    print(model.most_similar('A'))


if __name__ == '__main__':
    train()
    load_model()