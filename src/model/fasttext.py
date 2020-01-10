# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import config
from conf import model_config_fasttext as model_config
from utils.data_utils import get_pretrained_embedding


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if config.pretrain_embedding:
            embedding = get_pretrained_embedding()
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.num_vocab, config.embed_dim, padding_idx=config.padding_idx)
        self.embedding_ngram2 = nn.Embedding(model_config.n_gram_vocab, config.embed_dim)
        self.embedding_ngram3 = nn.Embedding(model_config.n_gram_vocab, config.embed_dim)
        self.dropout = nn.Dropout(model_config.dropout)
        self.fc1 = nn.Linear(config.embed_dim * 3, model_config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(model_config.hidden_size, config.num_classes)

    def forward(self, input_x, input_y):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
