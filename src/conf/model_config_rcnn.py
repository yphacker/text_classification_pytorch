# coding=utf-8
# author=yphacker

import os
from conf import config

model_save_path = os.path.join(config.model_path, 'rcnn.ckpt')
submission_path = os.path.join(config.data_path, 'rcnn_submission.csv')

embedding_pretrained =  config.embedding_pretrained
dropout = 0.5  # 随机失活
require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
learning_rate = 1e-3  # 学习率
embed_dim = embedding_pretrained.size(1) \
    if embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
hidden_size = 256  # lstm隐藏层
# num_layers = 1  # lstm层数
num_layers = 2
