# coding=utf-8
# author=yphacker

import gc
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from importlib import import_module
from conf import config
from utils.data_utils import load_vocab
from utils.model_utils import init_network, get_metrics

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, val_iter, criterion):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_x, batch_y in val_iter:
            batch_len = len(batch_y)
            data_len += batch_len
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item() * batch_len
            y_true_list += batch_y.cpu().numpy().tolist()
            y_pred_list += probs.cpu().numpy().tolist()
    # print('val metrics')
    # get_metrics(np.array(y_true_list), np.array(y_pred_list))
    return total_loss / data_len, roc_auc_score(y_true_list, y_pred_list)


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data, device)
    val_dataset = MyDataset(val_data, device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = x.Model().to(device)
    if model_name not in ['transformer', 'bert', 'albert']:
        init_network(model)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    y_true_list = []
    y_pred_list = []
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)

            train_loss = criterion(probs, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            y_true_list += batch_y.cpu().numpy().tolist()
            y_pred_list += probs.cpu().numpy().tolist()
            if cur_step % config.train_print_step == 0:
                train_score = roc_auc_score(y_true_list, y_pred_list)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train score: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), train_score))
                y_true_list = []
                y_pred_list = []
        val_loss, val_score = evaluate(model, val_loader, criterion)
        if val_score >= best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        # msg = 'the current epoch: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%},  ' \
        #       'val loss: {4:>5.2}, val acc: {5:>6.2%}, {6}'
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_score,
                         end_time - start_time, improved_str))

        if cur_epoch - last_improved_epoch > config.patience_epoch:
            if adjust_lr_num >= model_config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            scheduler.step()
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1
    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_score


def eval():
    pass


def predict():
    model = x.Model().to(device)
    model.load_state_dict(torch.load(model_config.model_save_path))
    model.eval()
    test_df = pd.read_csv(config.test_path)
    data_len = test_df.shape[0]
    submission = pd.read_csv(config.sample_submission_path)

    test_dataset = MyDataset(test_df, device)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch_x, _ in test_iter:
            prob = model(batch_x)
            # predictions.append(pred_y.cpu().detach().numpy().tolist())
            predictions.extend(prob.cpu().numpy())
            # submission.iloc[start_id: end_id][columns] = y_pred.cpu().numpy()
    submission[config.columns] = predictions
    submission.to_csv(model_config.submission_path, index=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        if args.mode == 1:
            x = train_df['comment_text'].values
            # y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
            y = train_df['toxic'].values
            skf = StratifiedKFold(n_splits=config.n_splits, random_state=0, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            score = 0
            score_list = []
            for fold_idx in range(config.n_splits):
                score += model_score[fold_idx]
                score_list.append('{:.4f}'.format(model_score[fold_idx]))
            print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
        else:

            # train_df = train_df[:50]
            train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--model", default='cnn', type=str, required=True,
                        help="choose a model: cnn, rnn, rcnn, rnn_atten, dpcnn, transformer, bert")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    # parser.add_argument('-em', '--embedding', default='pre_trained', type=str, help='random or pre_trained')
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model

    x = import_module('model.{}'.format(model_name))
    model_config = import_module('conf.model_config_{}'.format(model_name))

    model_config.submission_path = os.path.join(config.data_path, '{}_submission.csv'.format(model_name))

    if model_name == 'bert':
        from utils.bert_data_utils import MyDataset
    elif model_name == 'albert':
        from utils.albert_data_utils import MyDataset
    else:
        from utils.data_utils import MyDataset

    word2idx, idx2word = load_vocab()
    model_score = dict()
    main(args.operation)
