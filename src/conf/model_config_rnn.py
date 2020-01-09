# coding=utf-8
# author=yphacker

import os
from conf import config


model_save_path = os.path.join(config.model_path, 'rnn.ckpt')
submission_path = os.path.join(config.data_path, 'rnn_submission.csv')

embedding_pretrained = config.embedding_pretrained
dropout = 0.5                                              # 随机失活
learning_rate = 1e-3                                       # 学习率
embed_dim = embedding_pretrained.size(1) \
    if embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
hidden_size = 128                                          # lstm隐藏层
num_layers = 2                                             # lstm层数

