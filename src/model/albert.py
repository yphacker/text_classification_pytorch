# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertTokenizer
from conf import config
from conf import model_config_albert as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = AlbertModel.from_pretrained(model_config.pretrain_model_path)
        self.tokenizer = AlbertTokenizer.from_pretrained(model_config.pretrain_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_x, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        attention_mask = (input_x != self.tokenizer.pad_token_id).float()
        outputs = self.bert(
            input_x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        cls_output = outputs[1]
        cls_output = self.classifier(cls_output)
        cls_output = torch.sigmoid(cls_output)
        return cls_output
