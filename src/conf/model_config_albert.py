# coding=utf-8
# author=yphacker

import os
from conf import config

pretrain_model_name = 'albert-base-v2'
# pretrain_model_name = 'albert-xxlarge-v2'
pretrain_model_path = os.path.join(config.pretrain_model_path, pretrain_model_name)
