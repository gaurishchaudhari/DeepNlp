from bunch import Bunch
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from evaluate import evaluate_model
from TextRNN import TextRNN
from utils import mini_batches

def train_model(X_train, y_train, X_val, y_val, word_embeddings):

    config = Bunch()
    config.vocab_size, config.embed_size = word_embeddings.shape
    config.batch_size = 8   #64
    config.lr = 0.01
    config.hidden_size = 32
    config.num_lstm_layers = 2
    config.dropout_keep = 0.8
    config.isBidirectional = True
    config.output_size = 1
    config.max_epochs = 10

    model = TextRNN(config, torch.tensor(word_embeddings, dtype=torch.float))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_op = nn.BCELoss()
    #loss_op = nn.NLLLoss()

    print('Number of Training Examples: ', len(X_train))
    print('Number of Validation Examples: ', len(X_val))

    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(config.max_epochs):
        batch_losses = []
        for batch_X, batch_y in mini_batches(X_train, y_train, config.batch_size):
            batch_X = torch.tensor(batch_X, dtype=torch.long)
            batch_y = torch.tensor(batch_y, dtype=torch.float)
            optimizer.zero_grad()
            pred_y = model.forward(batch_X)
            loss = loss_op(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        if epoch % 1 == 0:
            avg_loss = np.mean(batch_losses)
            val_acc = evaluate_model(model, X_val, y_val, config.batch_size)
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(val_acc)
            print('Epoch = {}, Loss = {}, Accuracy = {}%'.format(epoch + 1, round(avg_loss, 8), round(val_acc, 2)))

    return model, epoch_losses, epoch_accuracies
