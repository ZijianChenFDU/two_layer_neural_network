import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from trainer import Trainer
from random_grid_search import random_grid_search
from save_load import *
import pandas as pd

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 测试程序时，为加快速度，截取一部分训练数据
# x_train = x_train[:1000]
# t_train = t_train[:1000]

print("==== Hyper Parameter Search ======================================================")

# 超参数搜索
np.random.seed(2023)
best_hyper_params = random_grid_search([0.15,0.25],[-7,-4],[200,300],50,'random_grid_search_results.csv')

# 程序测试时少搜索几轮
# best_hyper_params = random_grid_search([0.1,0.2],[-7,-4],[200,300],10,'test_random_grid_search_results.csv')

# 程序测试时跳过调参
# best_hyper_params = {'lr':0.247521572190421, 'weight_decay':2.54659123942365e-07, 'hidden_size':292.0}

weight_decay_lambda = best_hyper_params['weight_decay']
lr = best_hyper_params['lr']
hidden_size = int(best_hyper_params['hidden_size'])

print("==== Training with the Optimal Hyper Parameters ==================================")
# 最优参数训练
network = TwoLayerNet(input_size=784, hidden_size_list=[hidden_size],
                      output_size=10, weight_decay_lambda=weight_decay_lambda)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=50, mini_batch_size=128,
                  optimizer_param={'lr': lr})

trainer.train()

save_network(trainer,'trainer.pkl')
save_network(trainer.network,'network.pkl')

# test_acc_list = trainer.test_acc_list[1:]
# train_acc_list = trainer.train_acc_list[1:]
# train_loss_list = trainer.train_loss_list[1:]
# test_loss_list = trainer.test_loss_list[1:]

test_acc_list_epoch = trainer.test_acc_list_epoch[1:]
train_acc_list_epoch = trainer.train_acc_list_epoch[1:]
train_loss_list_epoch = trainer.train_loss_list_epoch[1:]
test_loss_list_epoch = trainer.test_loss_list_epoch[1:]


# 绘制图形

# 按每个iteration画图
# x = np.arange(np.shape(train_loss_list)[0]+1)[1:]
# plt.plot(x, train_loss_list, 'g--', label='train', markevery=1)
# plt.plot(x, test_loss_list, 'k-', label='test', markevery=1)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.ylim(0, test_loss_list[0]+0.005)
# plt.legend(loc='best')
# # plt.savefig('loss.png', dpi = 512)
# plt.savefig('loss.svg')
# plt.show()

# x = np.arange(np.shape(train_acc_list)[0]+1)[1:]
# plt.plot(x, train_acc_list, 'g--', label='train', markevery=1)
# plt.plot(x, test_acc_list, 'k-', label='test', markevery=1)
# plt.xlabel("Iteration")
# plt.ylabel("Accuracy")
# plt.ylim(test_acc_list[0]-0.005, 1.005)
# plt.legend(loc='best')
# # plt.savefig('accuracy.png', dpi = 512)
# plt.savefig('accuracy.svg')
# plt.show()

# column_names = ['iter', 'train loss','test loss', 'train acc', 'test acc']
# loss_acc_df = pd.DataFrame(np.transpose([x,train_loss_list,test_loss_list,train_acc_list,test_acc_list]),
#                            columns=column_names)
# "\n---- Train and Test Loss and Accuracy per Iteration ------------------------------"
# print(loss_acc_df)
# loss_acc_df.to_csv('loss_acc.csv')


# 按每个epoch画图
x = np.arange(np.shape(train_loss_list_epoch)[0]+2)[2:]
plt.plot(x, train_loss_list_epoch, marker='o', label='train', markevery=1)
plt.plot(x, test_loss_list_epoch, marker='s', label='test', markevery=1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, test_loss_list_epoch[0]+0.005)
plt.legend(loc='best')
# plt.savefig('loss.png', dpi = 512)
plt.savefig('loss.svg')
plt.show()

x = np.arange(np.shape(train_acc_list_epoch)[0]+2)[2:]
plt.plot(x, train_acc_list_epoch, marker='o', label='train', markevery=1)
plt.plot(x, test_acc_list_epoch, marker='s', label='test', markevery=1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(test_acc_list_epoch[0]-0.005, 1.005)
plt.legend(loc='best')
# plt.savefig('accuracy.png', dpi = 512)
plt.savefig('accuracy.svg')
plt.show()

column_names = ['epoch','train loss','test loss', 'train acc', 'test acc']
loss_acc_df = pd.DataFrame(np.transpose([x,train_loss_list_epoch,test_loss_list_epoch,train_acc_list_epoch,test_acc_list_epoch]),
                           columns=column_names)
print("\n---- Train and Test Loss and Accuracy per Epoch ----------------------------------")
print(loss_acc_df)
loss_acc_df.to_csv('loss_acc_epoch_test.csv')

print("\n==== END =========================================================================")