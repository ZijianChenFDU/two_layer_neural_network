import numpy as np
from sgd import *
from tqdm import tqdm

class Trainer:
    """进行神经网络的训练的类"""
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=50, mini_batch_size=100, 
                 optimizer_param={'lr':0.01}, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        # SGD optimzer
        self.optimizer = SGD(**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = int(max(self.train_size / mini_batch_size, 1))
        self.max_iter = epochs * self.iter_per_epoch
        self.current_iter = 0
        self.current_epoch = 0
        
        if self.verbose:
            # 记录每个iteration结束时的loss和accuracy
            # self.train_loss_list = []
            # self.test_loss_list = []
            # self.train_acc_list = []
            # self.test_acc_list = []

            # 记录每个epoch结束时的loss和accuracy
            self.train_loss_list_epoch = []
            self.test_loss_list_epoch = []
            self.train_acc_list_epoch = []
            self.test_acc_list_epoch = []
        else:
            self.test_acc = 0

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        # train_loss = self.network.loss(x_batch, t_batch)
        # if self.verbose: print("train loss:" + str(train_loss))
        
        # if self.verbose:
        #     x_train_sample, t_train_sample = self.x_train, self.t_train
        #     x_test_sample, t_test_sample = self.x_test, self.t_test

        #     _train_loss = self.network.loss(x_train_sample, t_train_sample)
        #     _test_loss = self.network.loss(x_test_sample, t_test_sample)
        #     _train_acc = self.network.accuracy(x_train_sample, t_train_sample)
        #     _test_acc = self.network.accuracy(x_test_sample, t_test_sample)

        #     self.train_loss_list.append(_train_loss)
        #     self.train_acc_list.append(_train_acc)
        #     self.test_loss_list.append(_test_loss)
        #     self.test_acc_list.append(_test_acc)



        if self.current_iter % self.iter_per_epoch == 0:

            self.current_epoch += 1
            
            if self.verbose:
                x_train_sample, t_train_sample = self.x_train, self.t_train
                x_test_sample, t_test_sample = self.x_test, self.t_test
                
                _train_loss_epoch = self.network.loss(x_train_sample, t_train_sample)
                _test_loss_epoch = self.network.loss(x_test_sample, t_test_sample)
                _train_acc_epoch = self.network.accuracy(x_train_sample, t_train_sample)
                _test_acc_epoch = self.network.accuracy(x_test_sample, t_test_sample)

                self.train_loss_list_epoch.append(_train_loss_epoch)
                self.test_loss_list_epoch.append(_test_loss_epoch)
                self.train_acc_list_epoch.append(_train_acc_epoch)
                self.test_acc_list_epoch.append(_test_acc_epoch)

        self.current_iter += 1

    def train(self,return_param = False):
        max_iter = self.max_iter
        pbar = tqdm(range(max_iter))
        for _ in pbar:
            self.train_step()
            if self.verbose:
                pbar.set_description(" epoch %2d, train_acc %.4f, test_acc %.4f" % 
                                     (self.current_epoch, self.train_acc_list_epoch[-1], self.test_acc_list_epoch[-1]))


        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if not self.verbose:
            self.test_acc = test_acc

        if self.verbose:
            print("==== Final Test Accuracy =========================================================")
            print(" test acc: " + str(test_acc))
            
        if return_param == True:
            return self.network.params
