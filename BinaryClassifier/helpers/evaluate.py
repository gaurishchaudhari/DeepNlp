import torch
import numpy as np
from helpers.utils import mini_batches

def evaluate_model(model, X, y, X_seqlen, batch_size, num_classes=2):
    all_pred_y = []
    all_y = []
    batch_losses = []

    with torch.no_grad():
        model.eval()
        for batch_X, batch_y, batch_X_seqlen in mini_batches(X, y, X_seqlen, batch_size):
            batch_X = torch.from_numpy(np.asarray(batch_X)).type(torch.LongTensor)
            batch_y = torch.from_numpy(np.asarray(batch_y)).type(torch.FloatTensor)
            batch_X_seqlen = torch.from_numpy(np.asarray(batch_X_seqlen)).type(torch.LongTensor)

            pred_y = model(batch_X, batch_X_seqlen)
            loss = model.loss_op(pred_y, batch_y)
            batch_losses.append(loss.item())
            if num_classes == 2:
                pred_y = [1 if x >= 0.5 else 0 for x in pred_y.numpy()]
            else:
                pred_y = torch.max(pred_y, axis=1)
            batch_y = [int(y) for y in np.squeeze(batch_y)]
            all_pred_y.extend(pred_y)
            all_y.extend(batch_y)

        correct = np.sum(np.asarray(all_pred_y) == np.asarray(all_y))
        return 100 * correct / len(all_y), correct, len(all_y), np.mean(batch_losses)


def predict_instance(model, ds, text, num_classes = 2):

    with torch.no_grad():
        X = [ds.get_X_repr(text)]
        X = torch.tensor(X, dtype=torch.long)
        X_len = torch.tensor([len(X)], dtype=torch.long)
        y = model(X, X_len)

        if num_classes == 2:
            return 1 if y.numpy() >= 0.5 else 0, np.squeeze(y.numpy())
        else:
            return torch.max(y).numpy(), np.squeeze(y.numpy())