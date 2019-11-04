import re
import numpy as np

def get_glove_vocab(glove_file):
    glove_vocab = set()
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0].lower()
            glove_vocab.add(word)
    print('Glove Vocab Size: ', len(glove_vocab))
    return glove_vocab


def load_word_embeddings(glove_file, word2index, index2word, store_trimmed = False):
    word_embeddings = [None for _ in range(len(word2index))]
    avg_vec = np.array([])
    num_vec = 0
    num_known_words = 0
    num_unknown_words = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0].lower()
            embedding = np.array([float(v) for v in parts[1:]])
            if num_vec == 0:
                avg_vec = embedding
                num_vec = 1
            else:
                avg_vec = (1.0 / (num_vec + 1)) * (num_vec * avg_vec + embedding)
                num_vec += 1
            if word in word2index:
                id = word2index[word]
                word_embeddings[id] = embedding
                num_known_words += 1

    if store_trimmed:
        with open(glove_file + '.trimmed', 'w', encoding='utf-8') as fout:
            for i in range(len(word_embeddings)):
                w = word_embeddings[i]
                if w is not None:
                    fout.write('{} {}\n'.format(index2word[i], ' '.join([str(v) for v in w])))

    for i in range(len(word_embeddings)):
        if i == 0:
            word_embeddings[0] = [0.0] * len(avg_vec)   # padded token
            continue
        if i == 1:
            word_embeddings[1] = avg_vec    # $UNK$ token
            continue

        if word_embeddings[i] is None:
            word_embeddings[i] = avg_vec
            num_unknown_words += 1

    word_embeddings = np.array(word_embeddings)
    word_embeddings = word_embeddings.reshape((-1, len(avg_vec)))
    print('shape(word_embddings)', word_embeddings.shape)

    print('Total number of KNOWN words: ', num_known_words)
    print('Total number of UNKNOWN words: ', num_unknown_words)

    return word_embeddings


def mini_batches(X, y, batch_size):
    X_batch = []
    y_batch = []
    for i in range(len(X)):
        if len(X_batch) == batch_size:
            yield X_batch, y_batch
            X_batch = []
            y_batch = []

        X_batch.append(X[i])
        y_batch.append(y[i])
    if len(X_batch) > 0:
        yield X_batch, y_batch