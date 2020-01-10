# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config
from conf import model_config_rcnn as model_config
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
        self.maxpool = nn.MaxPool1d(config.max_seq_len)
        self.fc = nn.Linear(model_config.hidden_size * 2 + config.embed_dim, config.num_classes)

    def forward(self, input_x):
        embed = self.embedding(input_x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        pred_y = torch.sigmoid(out)
        return pred_y
