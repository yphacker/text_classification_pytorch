# coding=utf-8
# author=yphacker

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import train_test_split
from importlib import import_module
from conf import config
from utils.data_utils import load_vocab
from utils.model_utils import init_network, get_loss, get_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, val_iter):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_x, batch_y in val_iter:
            batch_len = len(batch_y)
            data_len += batch_len

            pred_y = model(batch_x)
            loss = get_loss(pred_y, batch_y)
            total_loss += loss.item() * batch_len
            y_true_list += batch_y.cpu().numpy().tolist()
            y_pred_list += pred_y.cpu().numpy().tolist()
    # print('val metrics')
    # get_metrics(np.array(y_true_list), np.array(y_pred_list))
    return total_loss / data_len


def train():
    train_df = pd.read_csv(config.train_path)
    # train_df = train_df[:50]
    train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
    print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))

    train_dataset = MyDataset(train_data, device)
    val_dataset = MyDataset(val_data, device)

    train_iter = DataLoader(train_dataset, batch_size=config.batch_size)
    val_iter = DataLoader(val_dataset, batch_size=config.batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    # optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays
    warmup_steps = 10 ** 3
    # total_steps = len(train_iter) * config.epochs_num - warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # x_train, y_train = load_data(config.train_path)
    # x_val, y_val = load_data(config.val_path)
    # print('train:{}, val:{}'.format(len(y_train), len(y_val)))
    #
    # x_train, y_train = encode_data(x_train, y_train, word2idx)
    # x_val, y_val = encode_data(x_val, y_val, word2idx)

    cur_step = 1
    total_step = len(train_iter) * config.epochs_num
    last_improved_step = 0
    best_val_loss = 100
    flag = False
    for epoch in range(config.epochs_num):
        for batch_x, batch_y in train_iter:
            print(batch_x, batch_y)
            model.train()
            optimizer.zero_grad()
            pred_y = model(batch_x)
            train_loss = get_loss(pred_y, batch_y)
            # import torch.nn as nn
            # criterion = nn.BCELoss()
            # train_loss = criterion(pred_y, batch_y)
            train_loss.backward()
            optimizer.step()
            cur_step += 1
            if cur_step % config.print_per_batch == 0:
                val_loss = evaluate(model, val_iter)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_config.model_save_path)
                    improved_str = '*'
                    last_improved_step = cur_step
                else:
                    improved_str = ''
                    scheduler.step()
                # msg = 'the current step: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  ' \
                #       'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                # print(msg.format(cur_step, loss.item(), train_acc, dev_loss, dev_acc, improve))
                msg = 'the current step:{0}/{1}, train loss: {2:>5.2}, val loss: {3:>5.2}, {4}'
                print(msg.format(cur_step, total_step, train_loss.item(), val_loss, improved_str))
            print(cur_step, last_improved_step, len(train_iter))
            if cur_step - last_improved_step > len(train_iter):
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def eval():
    pass


def predict():
    model.load_state_dict(torch.load(model_config.model_save_path))
    model.eval()
    test_df = pd.read_csv(config.test_path)
    # test_df = test_df[:5]
    data_len = test_df.shape[0]
    num_batch = int((data_len - 1) / config.batch_size) + 1
    submission = pd.read_csv(config.sample_submission_path)
    # submission = submission[:5]
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    test_dataset = MyDataset(test_df, device)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch_x, _ in test_iter:
            pred_y = model(batch_x)
            # predictions.append(pred_y.cpu().detach().numpy().tolist())
            predictions.extend(pred_y.cpu().numpy())
            # submission.iloc[start_id: end_id][columns] = y_pred.cpu().numpy()
    submission[columns] = predictions
    submission.to_csv(model_config.submission_path, index=False)


def main(op):
    if op == 'train':
        train()
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

    # parser.add_argument('-em', '--embedding', default='pre_trained', type=str, help='random or pre_trained')
    # parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model

    x = import_module('model.{}'.format(model_name))
    model_config = import_module('conf.model_config_{}'.format(model_name))

    model_config.model_save_path = os.path.join(config.model_path, '{}.ckpt'.format(model_name))
    model_config.submission_path = os.path.join(config.data_path, '{}_submission.csv'.format(model_name))
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    model = x.Model().to(device)
    if model_name == 'bert':
        from utils.bert_data_utils import MyDataset
    elif model_name == 'albert':
        from utils.albert_data_utils import MyDataset
    else:
        from utils.data_utils import MyDataset
    if model_name not in ['transformer', 'bert', 'albert']:
        init_network(model)

    word2idx, idx2word = load_vocab()
    op = args.operation
    main(op)
