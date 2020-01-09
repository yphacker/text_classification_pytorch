# coding=utf-8
# author=yphacker


import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config
from conf import model_config_cnn as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if config.pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding_path, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_vocab, model_config.embed_dim, padding_idx=config.padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, model_config.num_filters, (k, model_config.embed_dim)) for k in model_config.filter_sizes])
        self.dropout = nn.Dropout(model_config.dropout)
        self.fc = nn.Linear(model_config.num_filters * len(model_config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        # x: (batch, 1, sentence_length,  )
        x = F.relu(conv(x)).squeeze(3)
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch, kernel_num)
        return x

    def forward(self, input_x):
        # x: (batch, seq_len)
        x = self.embedding(input_x)
        # x: (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)
        # x: (batch, 1, seq_len, embed_dim)
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        pred_y = torch.sigmoid(out)
        return pred_y
