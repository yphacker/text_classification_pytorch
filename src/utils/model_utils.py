# coding=utf-8
# author=yphacker

import torch.nn as nn
from sklearn.metrics import roc_auc_score
from conf import config


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=0):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def get_score(y_true, y_pred):
    score = 0
    for i, label in enumerate(config.label_columns):
        try:
            # print('{} roc_auc: {}'.format(label, roc_auc_score(y_true[:, i], y_pred[:, i])))
            score += roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            continue
    return score / len(config.label_columns)
