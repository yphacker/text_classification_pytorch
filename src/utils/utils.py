# coding=utf-8
# author=yphacker

import time
from datetime import timedelta
import numpy as np
import torch


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


if __name__ == "__main__":
    pass
