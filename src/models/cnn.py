# coding=utf-8
# author=yphacker


import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config
from conf import model_config_cnn as model_config
from utils.data_utils import get_pretrain_embedding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if config.pretrain_embedding:
            embedding = get_pretrain_embedding()
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_vocab, config.embed_dim, padding_idx=config.padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, model_config.num_filters, (k, config.embed_dim)) for k in model_config.filter_sizes])
        self.dropout = nn.Dropout(model_config.dropout)
        self.classifier = nn.Linear(model_config.num_filters * len(model_config.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        # x: (batch, 1, sentence_length,  )
        x = F.relu(conv(x)).squeeze(3)
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch, kernel_num)
        return x

    def forward(self, input_ids):
        # x: (batch, seq_len)
        x = self.embedding(input_ids)
        # x: (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)
        # x: (batch, 1, seq_len, embed_dim)
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.classifier(out)
        # pred_y = torch.sigmoid(out)
        return logits
