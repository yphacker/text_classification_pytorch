# coding=utf-8
# author=yphacker

import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from importlib import import_module
from conf import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, test_loader):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            # compute output
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            pred_list += [_.item() for _ in probs]
    return pred_list


def model_predict(model_name):
    if model_name in ['bert', 'albert', 'xlmroberta']:
        from utils.data_utils_plus import MyDataset
    else:
        from utils.data_utils import MyDataset

    test_dataset = MyDataset(test_df, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    preds_dict = dict()
    for fold_idx in range(config.n_splits):
        model = x.Model().to(device)
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))
        model.load_state_dict(torch.load(model_save_path))
        pred_list = predict(model, test_loader)
        submission = pd.DataFrame(pred_list)
        submission.to_csv('{}/{}_fold{}_submission.csv'
                          .format(config.submission_path, model_name, fold_idx), index=False, header=False)
        preds_dict['{}_{}'.format(model_name, fold_idx)] = pred_list
    pred_list = get_pred_list(preds_dict)

    submission = pd.read_csv(config.sample_submission_path)
    submission[config.label_columns] = pred_list
    submission.to_csv('submission.csv', index=False)


def file2submission():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            df = pd.read_csv('{}/{}_fold{}_submission.csv'
                             .format(config.submission_path, model_name, fold_idx), header=None)
            preds_dict['{}_{}'.format(model_name, fold_idx)] = df.values
    pred_list = get_pred_list(preds_dict)

    submission = pd.read_csv(config.sample_submission_path)
    submission[config.label_columns] = pred_list
    submission.to_csv('submission.csv', index=False)


def get_pred_list(preds_dict):
    pred_list = []
    for i in range(data_len):
        prob = None
        for model_name in model_name_list:
            for fold_idx in range(config.n_splits):
                if prob is None:
                    prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i] * ratio_dict[model_name]
                else:
                    prob += preds_dict['{}_{}'.format(model_name, fold_idx)][i] * ratio_dict[model_name]
        prob = prob / config.n_splits
        pred_list.append(prob)
    return pred_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-m", "--model_names", default='cnn', type=str, help="cnn")
    parser.add_argument("-type", "--pred_type", default='model', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="1:加权融合，2:投票融合")
    parser.add_argument("-r", "--ratios", default='1', type=str, help="融合比例")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    model_name_list = args.model_names.split('+')
    ratio_dict = dict()
    ratios = args.ratios
    ratio_list = args.ratios.split(',')
    for i, ratio in enumerate(ratio_list):
        ratio_dict[model_name_list[i]] = int(ratio)
    mode = args.mode

    test_df = pd.read_csv(config.test_path)
    data_len = test_df.shape[0]

    if args.pred_type == 'model':
        model_name = args.model_names
        x = import_module('model.{}'.format(model_name))
        model_predict(model_name)
    elif args.pred_type == 'file':
        file2submission()
