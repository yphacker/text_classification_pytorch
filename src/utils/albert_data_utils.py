# coding=utf-8
# author=yphacker


import torch
from torch.utils.data import Dataset
from transformers import AlbertTokenizer
from conf import config
from conf import model_config_albert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):

    def __init__(self, df, mode='train'):
        self.mode=mode
        self.tokenizer = AlbertTokenizer.from_pretrained(model_config_albert.pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.device = device
        self.x_data = []
        self.y_data = []
        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.x_data.append(x)
            self.y_data.append(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def row_to_tensor(self, tokenizer, row):
        x_data = row["comment_text"]
        x_encode = tokenizer.encode(x_data, max_length=config.max_seq_len)
        padding = [0] * (config.max_seq_len - len(x_encode))
        x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        if self.mode == 'test':
            y_tensor = torch.tensor([0] * len(config.label_columns), dtype=torch.float32)
        else:
            y_data = row[config.label_columns]
            y_tensor = torch.tensor(y_data, dtype=torch.float32)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


if __name__ == "__main__":
    pass
