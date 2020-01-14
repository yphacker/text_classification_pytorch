# coding=utf-8
# author=yphacker

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from conf import config
from conf import model_config_bert as model_config



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_config.pretrain_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_config.pretrain_model_path)
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

# class Model(nn.Module):
#
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.fc = nn.Linear(config.hidden_size, config.num_classes)
#
#     def forward(self, x):
#         context = x[0]  # 输入的句子
#         mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
#         _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
#         out = self.fc(pooled)
#         return out
