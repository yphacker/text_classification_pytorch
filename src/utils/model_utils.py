# coding=utf-8
# author=yphacker

import torch.nn as nn
from sklearn.metrics import roc_auc_score


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


# def get_loss_and_metrics(y_true, y_pred):
#     criterion = nn.BCELoss()
#     loss = criterion(y_pred, y_true)
#     roc_list = []
#     for i, name in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
#         # print(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
#         print(y_true[:, i].cpu().numpy().tolist())
#         print(y_pred[:, i].cpu().detach().numpy().tolist())
#         roc_list.append(roc_auc_score(y_true[:, i].cpu().numpy().tolist(), y_pred[:, i].cpu().detach().numpy().tolist()))
#     return loss, roc_list

def get_loss(y_pred, y_true):
    criterion = nn.BCELoss()
    loss = criterion(y_pred, y_true)
    return loss


def get_metrics(y_true, y_pred):
    for i, label in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        try:
            print('{} roc_auc: {}'.format(label, roc_auc_score(y_true[:, i], y_pred[:, i])))
        except:
            continue
