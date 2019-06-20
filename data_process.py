# -*- coding: utf-8 -*-
# @Time    : 2019/6/17 0:18
# @Author  : Tianchiyue
# @File    : data_process.py
# @Software: PyCharm
import numpy as np
from utils import *
import os

def load_X2id(file_path):
    X2id = {}
    id2X = {}
    with open(file_path) as f:
        for line in f:
            temp = line.strip().split()
            id, X = temp[0], temp[1]
            X2id[X] = int(id)
            id2X[int(id)] = X
    return X2id, id2X


def load_word2vec(file_path):
    word2vec = {}
    with open(file_path) as lines:
        for line in lines:
            split = line.split(" ")
            word = split[0]
            vector_strings = split[1:]
            vector = [float(num) for num in vector_strings]
            word2vec[word] = np.array(vector)
    return word2vec


def create_id2vec(word2id, word2vec):
    unk_vec = word2vec["unk"]
    dim_of_vector = len(unk_vec)
    num_of_tokens = len(word2id)+1
    id2vec = np.zeros((num_of_tokens, dim_of_vector))
    for word, t_id in word2id.items():
        id2vec[t_id, :] = word2vec[word] if word in word2vec else unk_vec
    return id2vec


def write_local(file_path, X2freq, start_idx=0):
    with open(file_path, "w") as f:
        for i, (X, freq) in enumerate(sorted(X2freq.items(), key=lambda t: -t[1]), start_idx):
            f.write(str(i) + "\t" + X + "\t" + str(freq) + "\n")


def write_data(corpus_path, write_pickle_path):
    num_of_labels = len(label2id)
    windows = 10
    all_data = []
    with open(corpus_path) as f:
        cnt = 0
        for line in f:
            if len(line.split("\t")) != 5:
                continue
            (start, end, words, labels, features) = line.strip().split("\t")

            labels, words, features = labels.split(), words.split(), features.split()
            length = len(words)
            start, end = int(start), int(end)
            labels_code = [0 for i in range(num_of_labels)]
            for label in labels:
                if label in label2id:
                    labels_code[label2id[label]] = 1

            left_start = 0 if (int(start) - windows) < 0 else int(start) - windows
            right_end = int(start) + windows
            left_content = words[left_start:start]
            entity = words[start:end]
            right_content = words[end:right_end]
            #  给一个标记<START>,<END> 防止pad
            if len(left_content) < 1:
                left_content = ["<LEFT>"]
            if len(right_content) < 1:
                right_content = ['<RIGHT>']
            if len(entity) < 1:
                continue
            left_wordid = tokenizer.text_to_sequence(left_content)
            entity_wordid = tokenizer.text_to_sequence(entity)
            right_wordid = tokenizer.text_to_sequence(right_content, reverse=True)

            features_code = [feature2id[feature] for feature in features]
            all_data.append((left_wordid, entity_wordid, right_wordid, features_code, labels_code))
            cnt += 1
            if cnt % 100000 == 0:
                print(cnt)
    x0 = np.array([x[0] for x in all_data])
    x1 = np.array([x[1] for x in all_data])
    x2 = np.array([x[2] for x in all_data])
    x3 = np.array([x[3] for x in all_data])
    y = np.array([x[4] for x in all_data])
    write_pickle(write_pickle_path,(x0, x1, x2, y))


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def load(self, word2idx):
        self.word2idx = word2idx

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, words, reverse=False, padding='post', truncating='post'):
        #         if self.lower:
        #             text = text.lower()
        #         words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


if __name__ == '__main__':
    if not os.path.exists('processed'):
        os.mkdir('processed/')
    raw_data_file = 'data/Wiki/all.txt'
    word_freq_path = 'processed/word2id_figer.txt '
    feature_freq_path = 'processed/feature2id_figer.txt'
    label_freq_path = 'processed/label2id_figer.txt'
    word_vec_path = '../../vector/glove.840B.300d.txt'
    freq = False
    if freq:
        word2freq = {}
        feature2freq = {}
        label2freq = {}
        with open(raw_data_file) as f:
            for line in f:
                temp = line.strip().split("\t")
                labels, features, words = temp[3], temp[4], temp[2]
                for label in labels.split():
                    if label not in label2freq:
                        label2freq[label] = 1
                    else:
                        label2freq[label] += 1
                for word in words.split():
                    if word not in word2freq:
                        word2freq[word] = 1
                    else:
                        word2freq[word] += 1
                for feature in features.split():
                    if feature not in feature2freq:
                        feature2freq[feature] = 1
                    else:
                        feature2freq[feature] += 1
        write_local(word_freq_path, word2freq, start_idx=1)
        write_local(feature_freq_path, feature2freq, start_idx=1)
        write_local(label_freq_path, label2freq)
        print("freq ok")
    # print "word2id..."
    word2id, id2word = load_X2id(word_freq_path)
    # print "feature2id..."
    feature2id, id2feature = load_X2id(feature_freq_path)
    # print "label2id..."
    label2id, id2label = load_X2id(label_freq_path)

    # print "word2vec..."
    word2vec = load_word2vec(word_vec_path)
    # print "id2vec..."
    id2vec = create_id2vec(word2id, word2vec)
    dicts = {"id2vec": id2vec, "word2id": word2id, "id2word": id2word, "label2id": label2id, "id2label": id2label,
             "feature2id": feature2id, "id2feature": id2feature}
    write_pickle('processed/dicts_figer.pkl',dicts)
    print('id ok')
    word2id["<LEFT>"] = len(word2id) + 1
    word2id["<RIGHT>"] = len(word2id) + 1

    tokenizer = Tokenizer(10, lower=False)
    tokenizer.load(word2id)
    embedding_matrix = np.concatenate(
        (np.array([id2vec[i] for i in range(len(id2vec))]), np.zeros((2, 300))), axis=0)
    write_pickle('processed/embedding_matrix.pkl', embedding_matrix)
    write_data('data/Wiki/train.txt', 'processed/train_figer.pkl')
    print('train ok')
    write_data('data/Wiki/dev.txt', 'processed/dev_figer.pkl')
    print('dev ok')
    write_data('data/Wiki/test.txt', 'processed/test_figer.pkl')
    print('test ok')