import torch
import numpy as np

def evaluate_model(model, X, y, X_seqlen, batch_size, num_classes=2):

    set_batch_X = np.array_split(X, len(X) / batch_size)
    set_batch_y = np.array_split(y, len(y) / batch_size)
    set_batch_X_seqlen = np.array_split(X_seqlen, len(X_seqlen) / batch_size)

    all_pred_y = []
    all_y = []
    with torch.no_grad():
        model.eval()
        for idx in range(len(set_batch_X)):
            batch_X = torch.from_numpy(set_batch_X[idx]).type(torch.LongTensor)
            batch_y = torch.from_numpy(set_batch_y[idx]).type(torch.FloatTensor)
            batch_X_seqlen = torch.from_numpy(set_batch_X_seqlen[idx]).type(torch.LongTensor)

            pred_y = model(batch_X, batch_X_seqlen)
            if num_classes == 2:
                pred_y = [1 if x >= 0.5 else 0 for x in pred_y.numpy()]
            else:
                pred_y = torch.max(pred_y, axis=1)
            batch_y = [int(y) for y in np.squeeze(batch_y)]
            all_pred_y.extend(pred_y)
            all_y.extend(batch_y)

    correct = np.sum(np.asarray(all_pred_y) == np.asarray(all_y))
    return 100 * correct / len(all_y), correct, len(all_y)

def predict_instance(model, ds, text, num_classes = 2):
    with torch.no_grad():
        X = [ds.get_X_repr(text)]
        X = torch.tensor(X, dtype=torch.long)
        X_len = torch.tensor([len(X)], dtype=torch.long)
        y = model(X, X_len)
        if num_classes == 2:
            return 1 if y.numpy() >= 0.5 else 0
        else:
            return torch.max(y).numpy()