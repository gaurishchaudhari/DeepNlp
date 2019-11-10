import numpy as np

def mini_batches(X, y, X_seqlen, batch_size):
    X_batch = []
    y_batch = []
    X_seqlen_batch = []
    for i in range(len(X)):
        if len(X_batch) == batch_size:
            yield X_batch, y_batch, X_seqlen_batch
            X_batch = []
            y_batch = []
            X_seqlen_batch = []
        X_batch.append(X[i])
        y_batch.append(y[i])
        X_seqlen_batch.append(X_seqlen[i])
    if len(X_batch) > 0:
        yield X_batch, y_batch, X_seqlen_batch