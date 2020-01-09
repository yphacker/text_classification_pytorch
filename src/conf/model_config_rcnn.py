# coding=utf-8
# author=yphacker


dropout = 0.5  # 随机失活
require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
learning_rate = 1e-3  # 学习率
hidden_size = 256  # lstm隐藏层
# num_layers = 1  # lstm层数
num_layers = 2
