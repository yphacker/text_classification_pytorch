# coding=utf-8
# author=yphacker

import os
from conf import config


model_save_path = os.path.join(config.model_path, 'transformer.ckpt')
submission_path = os.path.join(config.data_path, 'transformer_submission.csv')

embedding_pretrained = config.embedding_pretrained
dropout = 0.5                                              # 随机失活
require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
n_vocab = 0                                                # 词表大小，在运行时赋值
learning_rate = 5e-4                                       # 学习率
embed_dim = embedding_pretrained.size(1) \
    if embedding_pretrained is not None else 300           # 字向量维度
dim_model = 300
hidden = 1024
last_hidden = 512
num_head = 5
num_encoder = 2