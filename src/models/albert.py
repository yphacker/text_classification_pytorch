# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import AlbertModel
from conf import config
from conf import model_config_albert as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = AlbertModel.from_pretrained(model_config.pretrain_model_path)
        self.config = self.model.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
