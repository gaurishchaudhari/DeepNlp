import torch
import torch.nn as nn
import numpy as np
from helpers.utils import mini_batches

class TextCNN(nn.Module):

    def __init__(self, config, word_embeddings, num_classes = 2):
        super(TextCNN, self).__init__()

        self.config = config

        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_size, padding_idx=0)  # V * d
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)  # Use PreTrained Glove

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=config.num_channels, kernel_size=(config.kernel_size[0], config.embed_size))
        self.pool1 = nn.MaxPool1d(config.max_seq_len - config.kernel_size[0] + 1)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=config.num_channels, kernel_size=(config.kernel_size[1], config.embed_size))
        self.pool2 = nn.MaxPool1d(config.max_seq_len - config.kernel_size[1] + 1)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=config.num_channels, kernel_size=(config.kernel_size[2], config.embed_size))
        self.pool3 = nn.MaxPool1d(config.max_seq_len - config.kernel_size[2] + 1)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(config.dropout_p)

        self.fc = nn.Linear(config.num_channels * len(config.kernel_size), config.output_size)

        self.softmax = nn.Sigmoid() if num_classes == 2 else nn.Softmax()


    def forward(self, x, x_len):
        # print('shape(x) = ', x.size()) # (batch_size, max_seq_len)

        embed_x = self.embeddings(x)
        # print('shape(embed_x) = ', embed_x.size()) # (batch_size, max_seqlen, embed_size)

        embed_x = embed_x.unsqueeze(1)
        # print('shape(embed_x) = ', embed_x.size()) # (batch_size, 1, max_seqlen, embed_size)

        conv1_out = self.conv1(embed_x)
        # print('shape(conv1_out) = ', conv1_out.size()) # (batch_size, num_channels, max_seqlen-kernel_size+1 ,1)
        conv1_act = self.relu(conv1_out.squeeze(3))
        # print('shape(conv1_act) = ', conv1_act.size()) # (batch_size, num_channels, max_seqlen-kernel_size+1)
        conv1_pool = self.pool1(conv1_act)
        # print('shape(conv1_pool) = ', conv1_pool.size()) # (batch_size, num_channels, 1)
        conv1_pool = conv1_pool.squeeze(2)
        # print('shape(conv1_pool) = ', conv1_pool.size()) # (batch_size, num_channels)

        conv2_out = self.conv2(embed_x)
        conv2_act = self.relu(conv2_out.squeeze(3))
        conv2_pool = self.pool2(conv2_act).squeeze(2)

        conv3_out = self.conv3(embed_x)
        conv3_act = self.relu(conv3_out.squeeze(3))
        conv3_pool = self.pool3(conv3_act).squeeze(2)

        feature_map = torch.cat([conv1_pool, conv2_pool, conv3_pool], dim=1)
        # print('shape(feature_map) = ', feature_map.size()) # (batch_size, num_kernels * num_channels)

        fc_in = self.dropout(feature_map)

        fc_out = self.fc(fc_in)

        return self.softmax(fc_out)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_op(self, loss_op):
        self.loss_op = loss_op

    def run_epoch(self, X, y, X_seqlen):
        batch_losses = []
        self.train()

        for batch_X, batch_y, batch_X_seqlen in mini_batches(X, y, X_seqlen, self.config.batch_size):
            batch_X = torch.from_numpy(np.asarray(batch_X)).type(torch.LongTensor)
            batch_y = torch.from_numpy(np.asarray(batch_y)).type(torch.FloatTensor)
            batch_X_seqlen = torch.from_numpy(np.asarray(batch_X_seqlen)).type(torch.LongTensor)

            self.optimizer.zero_grad()
            pred_y = self.forward(batch_X, batch_X_seqlen)
            loss = self.loss_op(pred_y, batch_y)
            loss.backward()
            self.optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses)
        return avg_loss


