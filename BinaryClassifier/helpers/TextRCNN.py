import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from helpers.utils import mini_batches

class TextRCNN(nn.Module):

    def __init__(self, config, word_embeddings):
        super(TextRCNN, self).__init__()

        self.config = config

        self.embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_size, padding_idx=0)   # V * d
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)   # Use PreTrained Glove

        self.lstm = nn.LSTM(input_size=config.embed_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_lstm_layers,
                            batch_first=True,   # input x is (batch, seqlen, dim_feature)
                            dropout=config.dropout_p,
                            bidirectional=config.isBidirectional)

        self.dropout = nn.Dropout(config.dropout_p)

        self.W = nn.Linear(config.embed_size + ((2 if config.isBidirectional else 1) * config.hidden_size),
                             config.conv_feature_dim)

        self.tanh = nn.Tanh()

        self.fc = nn.Linear(config.conv_feature_dim, config.output_size)

        self.softmax = nn.Sigmoid()


    def forward(self, x, x_seqlen):
        #print('shape(x) = ', x.size()) # (batch_size, max_seq_len)

        embed_x = self.embeddings(x)
        #print('shape(embed_x) = ', embed_x.size()) # (batch_size, max_seqlen, embed_size)

        lstm_out, (h_n, c_n) = self.lstm(embed_x)
        #print('shape(lstm_out) = ', lstm_out.size())  # (batch_size, max_seqlen, num_dir * hidden_size)
        #print('shape(h_n) = ', h_n.size())  # (num_layers * num_dir, batch_size, hidden_size)
        #print('shape(c_n) = ', c_n.size())  # (num_layers * num_dir, batch_size, hidden_size)

        feature_repr = torch.cat([lstm_out, embed_x], dim=2)
        #print('shape(feature_repr) = ', feature_repr.size())  # (batch_size, max_seqlen, num_dir * hidden_size + embed_size)

        conv_in = self.tanh(self.W(feature_repr))
        # print('shape(conv_in) = ', conv_in.size())  # (batch_size, max_seqlen, conv_feature_dim)
        conv_in_adj = conv_in.permute(0, 2, 1)
        # print('shape(conv_in_adj) = ', conv_in_adj.size())  # (batch_size, conv_feature_dim, max_seqlen)

        conv_out = F.max_pool1d(conv_in_adj, conv_in_adj.shape[2]).squeeze(2)
        # print('shape(conv_out) = ', conv_out.size())  # (batch_size, conv_feature_dim)

        feature_map = self.dropout(conv_out)
        #print('shape(feature_map) = ', feature_map.size())  # (batch_size, conv_feature_dim)

        fc_out = self.fc(feature_map)
        # print('shape(fc_out) = ', fc_out.size())  # (batch_size, output_size)

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