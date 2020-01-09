# coding=utf-8
# author=yphacker


dropout = 0.5  # 随机失活
require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
pad_size = 32  # 每句话处理成的长度(短填长切)
learning_rate = 1e-3  # 学习率

num_filters = 250  # 卷积核数量(channels数)
