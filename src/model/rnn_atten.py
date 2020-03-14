# coding=utf-8
# author=yphacker


import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config
from conf import model_config_rnn_atten as model_config
from utils.data_utils import get_pretrain_embedding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if config.pretrain_embedding:
            embedding = get_pretrain_embedding()
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_vocab, config.embed_dim, padding_idx=config.padding_idx)
        self.lstm = nn.LSTM(config.embed_dim, model_config.hidden_size, model_config.num_layers,
                            bidirectional=True, batch_first=True, dropout=model_config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.Tensor(model_config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(model_config.hidden_size * 2, model_config.hidden_size * 2)
        self.fc = nn.Linear(model_config.hidden_size2, config.num_labels)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)
        # x: [batch_size, seq_len, embed_dim]
        H, _ = self.lstm(x)
        # x: [batch_size, seq_len, hidden_size * num_direction]
        M = self.tanh1(H)
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        # x: [batch_size, hidden_size * num_direction]
        out = self.fc(out)
        pred_y = torch.sigmoid(out)
        return pred_y
