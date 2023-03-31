import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from trainer import Trainer
from sgd import SGD
from random_grid_search import random_grid_search
import pickle
from save_load import *

# 载入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 加载保存好的网络
network = load_network('network.pkl')

# 计算训练集和测试集的accuracy并输出结果
train_acc = network.accuracy(x_train, t_train)
test_acc = network.accuracy(x_test, t_test)


print("==== Test ========================================================================" + 
      "\n train_acc: " + str(train_acc) + "\n test_acc: " + str(test_acc))