# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config
from conf import model_config_dpcnn as model_config
from utils.data_utils import get_pretrain_embedding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if config.pretrain_embedding:
            embedding = get_pretrain_embedding()
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_vocab, config.embed_dim, padding_idx=config.padding_idx)
        self.conv_region = nn.Conv2d(1, model_config.num_filters, (3, config.embed_dim), stride=1)
        self.conv = nn.Conv2d(model_config.num_filters, model_config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(model_config.num_filters, config.num_classes)

    def forward(self, input_x):
        x = self.embedding(input_x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        pred_y = torch.sigmoid(x)
        return pred_y

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
