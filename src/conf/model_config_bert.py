# coding=utf-8
# author=yphacker

import os
from conf import config

# pretrain_model_name = 'bert-base-uncased'
pretrain_model_name = 'bert-large-uncased-whole-word-masking'
pretrain_model_path = os.path.join(config.pretrain_model_path, pretrain_model_name)