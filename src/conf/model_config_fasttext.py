# coding=utf-8
# author=yphacker

import os
from conf import config

model_save_path = os.path.join(config.model_path, 'fasttext.ckpt')
submission_path = os.path.join(config.data_path, 'fasttext_submission.csv')

embedding_pretrained = config.embedding_pretrained

dropout = 0.5  # 随机失活
require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
pad_size = 32  # 每句话处理成的长度(短填长切)
learning_rate = 1e-3  # 学习率
embed_dim = embedding_pretrained.size(1) \
    if embedding_pretrained is not None else 300  # 字向量维度
hidden_size = 256  # 隐藏层大小
n_gram_vocab = 250499  # ngram 词表大小
