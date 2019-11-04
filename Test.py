import numpy as np
def pad(X):
    max_seqlen = 5
    for i in range(len(X)):
        X[i] = X[i] + [0] * (max_seqlen - len(X[i]))

def mini_batches(X, y, batch_size = 32):
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

def run_test1():
    X = [[1,2,3], [1,2,3,4,5], [1,2], [1,2,3,4,5], [1]]
    pad(X)
    print(X)

def run_test2():
    X = [[1,2,3], [1,2,3,4,5,6], [1,2], [1,2,3,4,5], [1]]
    for bX, _ in mini_batches(X, X, 2):
        print(bX)

def run_test3():
    X = [[1,2,3], [5,2,3], [1,2,3]]
    avg = np.array([])
    num = 0
    X = np.array(X)
    for x in X:
        if num == 0:
            avg = x
            num = 1
        else:
            avg = (1 / (num + 1)) * np.add(num * avg, x)
            num += 1

    print(avg)


def print_word_embeddings():
    glove_file = 'C:\\Workspace\\Git\\Data\\glove.6B\\glove.6B.50d.txt'
    avg_vec = np.array([])
    num_vec = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            embedding = np.array([float(v) for v in parts[1:]])
            if num_vec == 0:
                avg_vec = embedding
                num_vec = 1
            else:
                avg_vec = (1.0 / (num_vec + 1)) * (num_vec * avg_vec + embedding)
                num_vec += 1

    print(avg_vec)


if __name__ == '__main__':
    print_word_embeddings()
    #run_test3()