#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:26:19 2019

@author: tom
"""
import numpy as np;
from visualize import plot_2, plot_3, plot_3d, plot_6;
from load_dataset import load_mnist_test, load_mnist_train;
from conduct_svm import svm;
from conduct_csc import nn_csc, csc, par_csc, par_nn_csc;
from pickless import output;

#%%実験のオプションを設定
def make_option():
    opt = {};
    # 訓練画像の枚数
    opt['train_amount'] = 500;
    # テスト画像の枚数
    opt['test_amount'] = 500;
    # フィルタのサイズ
    opt['d_size'] = [(5, 5, 6), (5, 5, 6, 16)];
    # スパースの尺度
    opt['lmbda'] = [0.5, 0.025];
    # 繰り返し回数
    opt['Iter'] = [200, 200];
    return opt;

#%%
opt = make_option();
train_data, train_label = load_mnist_train(opt["train_amount"]);
test_data, test_label = load_mnist_test(opt["test_amount"]);
d0, D0, coef0 = nn_csc(input_ = train_data, d_size = opt['d_size'][0], lmbda = opt['lmbda'][0], Iter = opt['Iter'][0], visualize = True);
d1, D1, coef1 = nn_csc(input_ = coef0.squeeze().transpose(0,1,3,2) , d_size = opt['d_size'][1], lmbda = opt['lmbda'][1], Iter = opt['Iter'][1], visualize = False);
output(opt, D0, coef0, D1, coef1);
#%%
print(np.sum(coef1<0))
#%%
# train_data, train_label = load_mnist_train(opt['train_amount']);
opt = make_option();
pd0, pD0, pcoef0 = par_nn_csc(input_ = train_data, d_size = opt['d_size'][0], lmbda = opt['lmbda'][0], Iter = opt['Iter'][0], visualize = True);
pd1, pD1, pcoef1 = par_nn_csc(input_ = coef0.squeeze().transpose(0,1,3,2) , d_size = opt['d_size'][1], lmbda = opt['lmbda'][1], Iter = opt['Iter'][1], visualize = False);