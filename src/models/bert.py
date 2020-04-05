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
        self.config = self.bert.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # outputs = torch.softmax(logits)
        return logits

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
