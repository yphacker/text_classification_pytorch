# coding=utf-8
# author=yphacker


import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data")
word_embedding_path = os.path.join(data_path, "glove.840B.300d.txt")
pretrain_model_path = os.path.join(data_path, "pretrain_model")
pretrain_embedding_path = os.path.join(data_path, "pretrain_embedding.npz")
model_path = os.path.join(data_path, "model")
vocab_path = os.path.join(data_path, "vocab.pkl")
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')
sample_submission_path = os.path.join(data_path, 'sample_submission.csv')

pretrain_embedding = False
# pretrain_embedding = True
embed_dim = 300
num_classes = 6
max_seq_len = 155

tokenizer = lambda x: x.split(' ')[:max_seq_len]
padding_idx = 0

num_vocab = 41530
batch_size = 32
epochs_num = 8
train_print_step = 20
patience_epoch = 4
