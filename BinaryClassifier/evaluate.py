import torch
import numpy as np
from utils import mini_batches


def evaluate_model(model, X, y, batch_size = 32):
    all_pred_y = []
    all_y = []
    with torch.no_grad():
        for batch_X, batch_y in mini_batches(X, y, batch_size):
            batch_X = torch.tensor(batch_X, dtype=torch.long)
            pred_y = model(batch_X)
            pred_y = pred_y.detach().numpy()
            pred_y = [1 if x >= 0.5 else 0 for x in pred_y]
            batch_y = np.squeeze(batch_y)
            all_pred_y.extend(pred_y)
            all_y.extend(batch_y)

    correct = np.dot(all_pred_y, all_y)
    return 100 * correct / len(all_y)


def predict_instance(model, ds, text):
    with torch.no_grad():
        X = [ds.get_X_repr(text)]
        X = torch.tensor(X, dtype=torch.long)
        y = model(X)
        #return 1 if np.squeeze(y.detach().numpy()) >= 0.5 else 0
        return y.detach().numpy()