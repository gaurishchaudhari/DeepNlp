import os
import re
import string
from bunch import Bunch
import numpy as np
from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self):
        self.raw_train = None
        self.raw_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_seqlen = None
        self.X_test_seqlen = None
        self.word2index = None
        self.index2word = None
        # tokenization regex
        self.regex = re.compile(r'[\s{}]+'.format(re.escape(string.punctuation)))
        self.vocabNotInGlove = None

    def load_dataset_from_dir(self, train_dir, test_dir, categories):
        self.raw_train = self.load_dataset_subset(train_dir, categories)
        self.raw_test = self.load_dataset_subset(test_dir, categories)

    def build_Xy_sequences(self):
        self.X_train, self.y_train = self.compute_Xy(self.raw_train)
        self.X_test, self.y_test = self.compute_Xy(self.raw_test)
        self.X_train_seqlen = [len(d) for d in self.X_train]
        self.X_test_seqlen = [len(d) for d in self.X_test]

    def pad_Xy_sequences(self):
        self.pad_X(self.X_train, self.X_train_seqlen)
        self.pad_X(self.X_test, self.X_test_seqlen)

    def to_numpy(self):
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train).reshape((-1, 1))
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test).reshape((-1, 1))

    def load_dataset_subset(self, data_dir, categories):
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
                    text = self.preprocess(text)
                    bch.data.append(text)
                bch.target.append(bch.target_names.index(c))
        # shuffle
        bch.data, bch.target = shuffle(bch.data, bch.target, random_state=42)
        return bch

    def preprocess(self, text):
        text = re.sub(r'<.*>?', '', text)   # remove html tags
        tokens = [t for t in self.regex.split(text.lower()) if t != '']
        #tokens = [t for t in tokens if self.has_letter(t)]   # keep tokens that have at least one letter
        return ' '.join(tokens)

    def has_letter(self, str):
        for c in str:
            if c.isalpha():
                return True
        return False

    def build_vocab(self, glove_vocab = None):
        self.word2index = {}
        self.index2word = {}
        self.vocabNotInGlove = set()

        # index 0 is reserved for Padded token
        # index 1 is reserved for $UNK$
        self.word2index['$PAD$'] = 0
        self.index2word[0] = '$PAD$'
        self.word2index['$UNK$'] = 1
        self.index2word[1] = '$UNK$'
        index = 2
        for text in self.raw_train.data:
            for token in text.split(' '):
                if token in self.word2index:
                    continue
                if glove_vocab is not None and token not in glove_vocab:
                    self.vocabNotInGlove.add(token)
                    continue
                self.word2index[token] = index
                self.index2word[index] = token
                index += 1

    def compute_Xy(self, raw_set):
        X_set = []
        y_set = []
        size = len(raw_set.data)
        for i in range(size):
            X_set.append(self.get_X_repr(raw_set.data[i]))
            y_set.append(raw_set.target[i])
        return X_set, y_set

    def get_X_repr(self, text):
        X = []
        for token in text.split(' '):
            idx = 1 # default is $UNK$
            if token in self.word2index:
                idx = self.word2index[token]
            X.append(idx)
        return X

    def pad_X(self, X, X_seqlen):
        max_seqlen = max(X_seqlen)
        for i in range(len(X)):
            X[i] = X[i] + [0] * max(max_seqlen - len(X[i]), 0)


def print_stats(ds):

    print('Number of Raw Training records: {} examples, {} labels '.format(
        len(ds.raw_train.data),
        len(ds.raw_train.target)))
    print('Number of Raw Test records: {} examples, {} labels '.format(
        len(ds.raw_test.data),
        len(ds.raw_test.target)))
    print('Number of X,y Training records: {} examples, {} labels '.format(len(ds.X_train), len(ds.y_train)))
    print('Number of X,y Test records: {} examples, {} labels '.format(len(ds.X_test), len(ds.y_test)))
    print('Target Names:', ds.raw_train.target_names)
    print('Training Set: Positives = {}, Negatives = {}'.format(
        len([l for l in ds.y_train if l == 1]),
        len([l for l in ds.y_train if l == 0])))
    print('Test Set: Positives = {}, Negatives = {}'.format(
        len([l for l in ds.y_test if l == 1]),
        len([l for l in ds.y_test if l == 0])))
    print('Vocab Size: ', len(ds.word2index))
    print('Number of words not in Glove: ', len(ds.vocabNotInGlove))
    print('Words not in Glove: ', list(ds.vocabNotInGlove)[:10])
    print('Top 5 Words in vocab', ['{}({})'.format(w, i) for w, i in ds.word2index.items() if i <= 5])
    print('Top 5 training examples:')
    for i in range(5):
        print('[#{}]\t {}'.format(i + 1, ds.y_train[i]))
        print('\t', ds.X_train[i][:10])
        print('\t >>', ds.raw_train.data[i][:100])
    print('X_train.shape = ', ds.X_train.shape)
    print('y_train.shape = ', ds.y_train.shape)
    print('X_test.shape = ', ds.X_test.shape)
    print('y_test.shape = ', ds.y_test.shape)
