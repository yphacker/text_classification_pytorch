# coding=utf-8
# author=yphacker

import os
import re
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from conf import config
from conf import model_config_bert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class FasttextDataset(Dataset):
#
#     def __init__(self, df, device='cpu'):
#         self.tokenizer = BertTokenizer.from_pretrained(model_config_bert.pretrain_model_path)
#         self.pad_idx = self.tokenizer.pad_token_id
#         self.device = device
#         self.x_data = []
#         self.y_data = []
#         for i, row in df.iterrows():
#             x, y = self.row_to_tensor(self.tokenizer, row)
#             self.x_data.append(x)
#             self.y_data.append(y)
#
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#
#     def __len__(self):
#         return len(self.y_data)
#
#     def row_to_tensor(self, tokenizer, row):
#         # x_data = row["comment_text"]
#         # y_data = row[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
#         # x_encode = tokenizer.encode(x_data, max_length=config.max_seq_len)
#         # padding = [0] * (config.max_seq_len - len(x_encode))
#         # x_encode += padding
#         # x_tensor = torch.tensor(x_encode, dtype=torch.long).to(self.device)
#         # y_tensor = torch.tensor(y_data, dtype=torch.float32).to(self.device)
#         # return x_tensor, y_tensor
#         x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
#         y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
#         bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#         trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)
#
#         # pad前的长度(超过pad_size的设为pad_size)
#         seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
#         return (x, seq_len, bigram, trigram), y
#
#     def biGramHash(sequence, t, buckets):
#         t1 = sequence[t - 1] if t - 1 >= 0 else 0
#         return (t1 * 14918087) % buckets
#
#     def triGramHash(sequence, t, buckets):
#         t1 = sequence[t - 1] if t - 1 >= 0 else 0
#         t2 = sequence[t - 2] if t - 2 >= 0 else 0
#         return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets
#
#     def load_dataset(path, pad_size=32):
#         contents = []
#         with open(path, 'r', encoding='UTF-8') as f:
#             for line in tqdm(f):
#                 lin = line.strip()
#                 if not lin:
#                     continue
#                 content, label = lin.split('\t')
#                 words_line = []
#                 token = tokenizer(content)
#                 seq_len = len(token)
#                 if pad_size:
#                     if len(token) < pad_size:
#                         token.extend([vocab.get(PAD)] * (pad_size - len(token)))
#                     else:
#                         token = token[:pad_size]
#                         seq_len = pad_size
#                 # word to id
#                 for word in token:
#                     words_line.append(vocab.get(word, vocab.get(UNK)))
#
#                 # fasttext ngram
#                 buckets = config.n_gram_vocab
#                 bigram = []
#                 trigram = []
#                 # ------ngram------
#                 for i in range(pad_size):
#                     bigram.append(biGramHash(words_line, i, buckets))
#                     trigram.append(triGramHash(words_line, i, buckets))
#                 # -----------------
#                 contents.append((words_line, int(label), seq_len, bigram, trigram))
#         return contents  # [([...], 0), ([...], 1), ...]


class MyDataset(Dataset):

    def __init__(self, df, device='cpu'):
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
        x_tensor = torch.tensor(x_encode, dtype=torch.long).to(self.device)
        label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        if label_columns[0] in row.index.tolist():
            y_data = row[label_columns]
            y_tensor = torch.tensor(y_data, dtype=torch.float32).to(self.device)
        else:
            y_tensor = torch.tensor([0] * len(label_columns), dtype=torch.float32).to(self.device)
        return x_tensor, y_tensor

    def encode(self, items, max_len=config.max_seq_len):
        x_idx = [self.pad_idx] * max_len
        for i, item in enumerate(items):
            if self.word2idx.get(item, None):
                x_idx[i] = self.word2idx[item]
        return x_idx

    def __len__(self):
        return len(self.y_data)


def encode_data(model_name, x_data=None, word2idx=dict()):
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_config_bert.pretrain_model_path)
        x_encode = tokenizer.encode(x_data, max_length=config.max_seq_len)
        padding = [0] * (config.max_seq_len - len(x_encode))
        x_encode += padding
    else:
        x_encode = [config.padding_idx] * config.max_seq_len
        x_data = clean_text(x_data)
        x_token = config.tokenizer(x_data)
        for i, word in enumerate(x_token):
            if word2idx.get(word, None):
                x_encode[i] = word2idx[word]
    x_tensor = torch.tensor(x_encode, dtype=torch.long).to(device)
    return x_tensor


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
    vocab_dic.update({'PAD': 0, 'UNK': 1})
    print(len(vocab_dic))
    pkl.dump(vocab_dic, open(config.vocab_path, 'wb'))
    return vocab_dic


def build_embedding_pretrained():
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    word2idx, idx2word = load_vocab()
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(config.word_embedding_path))

    embedding_matrix = np.zeros((config.num_vocab, config.embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.savez_compressed(config.pretrain_embedding_path, embeddings=embedding_matrix)


def get_embedding_pretrained():
    if config.pretrained_word_embedding:
        return torch.tensor(np.load(config.pretrain_embedding_path)["embeddings"].astype('float32'))
    else:
        return None


if __name__ == "__main__":
    # build_vocab()
    build_embedding_pretrained()
