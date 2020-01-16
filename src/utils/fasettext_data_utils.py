# coding=utf-8
# author=yphacker

# class FasttextDataset(Dataset):
#
#     def __init__(self, df, device='cpu'):
#         self.tokenizer = BertTokenizer.from_pretrained(model_config_bert.pretrain_model_path)
#         self.pad_idx = self.tokenizer.pad_token_id
#         self.device = device
#         self.x_data = []
#         self.y_data = []
#         for i, row in df.iterrows():
#             x, y = self.row_to_tensor(self.tokenizer, row)
#             self.x_data.append(x)
#             self.y_data.append(y)
#
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]
#
#     def __len__(self):
#         return len(self.y_data)
#
#     def row_to_tensor(self, tokenizer, row):
#         # x_data = row["comment_text"]
#         # y_data = row[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
#         # x_encode = tokenizer.encode(x_data, max_length=config.max_seq_len)
#         # padding = [0] * (config.max_seq_len - len(x_encode))
#         # x_encode += padding
#         # x_tensor = torch.tensor(x_encode, dtype=torch.long).to(self.device)
#         # y_tensor = torch.tensor(y_data, dtype=torch.float32).to(self.device)
#         # return x_tensor, y_tensor
#         x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
#         y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
#         bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
#         trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)
#
#         # pad前的长度(超过pad_size的设为pad_size)
#         seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
#         return (x, seq_len, bigram, trigram), y
#
#     def biGramHash(sequence, t, buckets):
#         t1 = sequence[t - 1] if t - 1 >= 0 else 0
#         return (t1 * 14918087) % buckets
#
#     def triGramHash(sequence, t, buckets):
#         t1 = sequence[t - 1] if t - 1 >= 0 else 0
#         t2 = sequence[t - 2] if t - 2 >= 0 else 0
#         return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets
#
#     def load_dataset(path, pad_size=32):
#         contents = []
#         with open(path, 'r', encoding='UTF-8') as f:
#             for line in tqdm(f):
#                 lin = line.strip()
#                 if not lin:
#                     continue
#                 content, label = lin.split('\t')
#                 words_line = []
#                 token = tokenizer(content)
#                 seq_len = len(token)
#                 if pad_size:
#                     if len(token) < pad_size:
#                         token.extend([vocab.get(PAD)] * (pad_size - len(token)))
#                     else:
#                         token = token[:pad_size]
#                         seq_len = pad_size
#                 # word to id
#                 for word in token:
#                     words_line.append(vocab.get(word, vocab.get(UNK)))
#
#                 # fasttext ngram
#                 buckets = config.n_gram_vocab
#                 bigram = []
#                 trigram = []
#                 # ------ngram------
#                 for i in range(pad_size):
#                     bigram.append(biGramHash(words_line, i, buckets))
#                     trigram.append(triGramHash(words_line, i, buckets))
#                 # -----------------
#                 contents.append((words_line, int(label), seq_len, bigram, trigram))
#         return contents  # [([...], 0), ([...], 1), ...]