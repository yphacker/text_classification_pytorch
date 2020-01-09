# coding=utf-8
# author=yphacker

import os
from conf import config


embedding_pretrained = config.embedding_pretrained
dropout = 0.5  # 随机失活
num_epochs = 20  # epoch数
# batch_size = 128  # mini-batch大小
pad_size = 32  # 每句话处理成的长度(短填长切)
learning_rate = 1e-3  # 学习率
embed_dim = embedding_pretrained.size(1) if embedding_pretrained is not None else 300  # 字向量维度
# filter_sizes = (3, 4, 5)  # 卷积核尺寸
filter_sizes = [3]  # 卷积核尺寸
# num_filters = 256  # 卷积核数量(channels数)
num_filters = 128  # 卷积核数量(channels数)
