from save_load import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

trainer = load_network('trainer.pkl')

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
plt.savefig('loss.png', dpi = 300)
plt.savefig('loss.svg')
plt.show()

x = np.arange(np.shape(train_acc_list_epoch)[0]+2)[2:]
plt.plot(x, train_acc_list_epoch, marker='o', label='train', markevery=1)
plt.plot(x, test_acc_list_epoch, marker='s', label='test', markevery=1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(test_acc_list_epoch[0]-0.005, 1.005)
plt.legend(loc='best')
plt.savefig('accuracy.png', dpi = 300)
plt.savefig('accuracy.svg')
plt.show()

column_names = ['epoch','train loss','test loss', 'train acc', 'test acc']
loss_acc_df = pd.DataFrame(np.transpose([x,train_loss_list_epoch,test_loss_list_epoch,train_acc_list_epoch,test_acc_list_epoch]),
                           columns=column_names)
print("\n---- Train and Test Loss and Accuracy per Epoch ----------------------------------")
print(loss_acc_df)
loss_acc_df.to_csv('loss_acc_epoch_test.csv')