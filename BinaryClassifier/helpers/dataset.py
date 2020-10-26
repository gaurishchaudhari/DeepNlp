import os
from bunch import Bunch
import numpy as np
from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, num_classes = 2):
        self.num_classes = num_classes
        self.raw_train = None
        self.raw_test = None
        self.vocab = None
        self.word2index = None
        self.index2word = None
        self.word_embeddings = None
        self.dim_embedding = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_seqlen = None
        self.X_test_seqlen = None

    def load_dataset_from_dir(self, train_dir, test_dir, categories, preprocess_text_func):
        self.raw_train = self.load_dataset_from_dir_subset(train_dir, categories, preprocess_text_func)
        self.raw_test = self.load_dataset_from_dir_subset(test_dir, categories, preprocess_text_func)

    def load_dataset_from_dir_subset(self, data_dir, categories, preprocess_text_func):
        bch = Bunch()
        bch.target_names = [c for c in categories]
        bch.data = []
        bch.target = []
        for c in categories:
            data_subset_dir = os.path.join(data_dir, c)
            for file in os.listdir(data_subset_dir):
                file_path = os.path.join(data_subset_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = preprocess_text_func(text)
                    bch.data.append(text)
                bch.target.append(bch.target_names.index(c))
        # shuffle
        bch.data, bch.target = shuffle(bch.data, bch.target, random_state=42)
        return bch

    def build_vocab(self):
        self.vocab = set()
        for text in self.raw_train.data:
            for token in text.split(' '):
                self.vocab.add(token)

    def load_word_embeddings(self, glove_file):
        self.word_embeddings = []
        self.word2index = {}
        self.index2word = {}

        # index 0 is reserved for Padded token
        # index 1 is reserved for $UNK$
        self.word2index['$PAD$'] = 0
        self.index2word[0] = '$PAD$'
        self.word_embeddings.append(None)

        self.word2index['$UNK$'] = 1
        self.index2word[1] = '$UNK$'
        self.word_embeddings.append(None)

        index = 2
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                word = parts[0].lower()
                if not word in self.vocab:
                    continue
                embedding = np.array([float(v) for v in parts[1:]])
                self.word2index[word] = index
                self.index2word[index] = word
                self.word_embeddings.append(embedding)
                index += 1

        if len(self.word_embeddings) > 2:
            self.dim_embedding = len(self.word_embeddings[2])
            self.word_embeddings[0] = np.zeros(self.dim_embedding)
            self.word_embeddings[1] = np.array(np.mean(self.word_embeddings[2:], axis=0))

        self.word_embeddings = np.asarray(self.word_embeddings)

    def compute_Xy(self):
        self.X_train, self.y_train = self.compute_Xy_subset(self.raw_train)
        self.X_test, self.y_test = self.compute_Xy_subset(self.raw_test)

    def compute_Xy_subset(self, raw_set):
        X_set = []
        y_set = []
        size = len(raw_set.data)
        for i in range(size):
            X_set.append(self.get_X_repr(raw_set.data[i]))
            y_set.append(self.get_Y_repr(raw_set.target[i], self.num_classes))
        return np.asarray(X_set), np.asarray(y_set).reshape((-1, 1) if self.num_classes == 2 else np.asarray(y_set))

    def get_X_repr(self, text):
        X = []
        for token in text.split(' '):
            idx = 1 # default is $UNK$
            if token in self.word2index:
                idx = self.word2index[token]
            X.append(idx)
        return np.asarray(X, dtype=np.int64)

    def get_Y_repr(self, label, num_classes):
        if num_classes == 2:
            return label
        else:
            Y = np.zeros(num_classes, dtype=np.int32)
            Y[label] = 1
            return Y

    def pad_Xy(self, fixed_max_len = -1):
        self.X_train_seqlen = [len((x if fixed_max_len == -1 else x[:fixed_max_len])) for x in self.X_train]
        self.X_test_seqlen = [len((x if fixed_max_len == -1 else x[:fixed_max_len])) for x in self.X_test]
        self.X_train = self.pad_X(self.X_train, self.X_train_seqlen)
        self.X_test = self.pad_X(self.X_test, self.X_test_seqlen)

    def pad_X(self, X, X_seqlen):
        max_seqlen = max(X_seqlen)
        pad_token = 0
        padded_X = np.ones((len(X), max_seqlen), dtype=np.int64) * pad_token
        for i, x_len in enumerate(X_seqlen):
            sequence = X[i]
            padded_X[i, 0:x_len] = sequence[0:x_len]
        return padded_X
