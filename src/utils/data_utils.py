# coding=utf-8
# author=yphacker

import os
import re
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from conf import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):

    def __init__(self, df, mode):
        self.mode = mode
        self.tokenizer = config.tokenizer
        self.pad_idx = config.padding_idx
        self.device = device
        self.x_data = []
        self.y_data = []
        self.word2idx, self.idx2word = load_vocab()
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def row_to_tensor(self, tokenizer, row):
        x_data = row["comment_text"]
        x_data = clean_text(x_data)
        x_token = tokenizer(x_data)
        x_encode = self.encode(x_token)
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        if self.mode == 'test':
            y_tensor = torch.tensor([0] * len(config.label_columns), dtype=torch.float32)
        else:
            y_data = row[config.label_columns]
            y_tensor = torch.tensor(y_data, dtype=torch.float32)

        return x_tensor, y_tensor

    def encode(self, items, max_len=config.max_seq_len):
        x_idx = [self.pad_idx] * max_len
        for i, item in enumerate(items):
            if self.word2idx.get(item, None):
                x_idx[i] = self.word2idx[item]
        return x_idx

    def __len__(self):
        return len(self.y_data)


def load_vocab():
    if os.path.exists(config.vocab_path):
        word2idx = pkl.load(open(config.vocab_path, 'rb'))
    else:
        word2idx = build_vocab()
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def clean_text(text):
    text = text.replace('\n', ' ').lower()
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = text.strip()
    return text


def build_vocab():
    tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    vocab_dic = {}
    vocab_max_size = 100000
    vocab_min_freq = 5

    train_df = pd.read_csv(config.train_path)
    texts = train_df['comment_text'].values.tolist()
    for text in texts:
        if not text:
            continue
        text = clean_text(text)
        for word in tokenizer(text):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= vocab_min_freq],
                        key=lambda x: x[1], reverse=True)[:vocab_max_size]
    vocab_dic = {word_count[0]: idx + 2 for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({'_PAD_': 0, '_UNK_': 1})
    print(len(vocab_dic))
    pkl.dump(vocab_dic, open(config.vocab_path, 'wb'))
    return vocab_dic


def build_embedding_pretrained():
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    word2idx, idx2word = load_vocab()
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(config.word_embedding_path))

    embedding_matrix = np.zeros((config.num_vocab, config.embed_dim))
    for word, i in word2idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.savez_compressed(config.pretrain_embedding_path, embeddings=embedding_matrix)


def get_pretrain_embedding():
    return torch.tensor(np.load(config.pretrain_embedding_path)["embeddings"].astype('float32'))


if __name__ == "__main__":
    # build_vocab()
    build_embedding_pretrained()
