#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:37:33 2020

@author: kuroki-lab
"""

import numpy as np;
import cv2;
from torchvision import datasets;
from sporco import plot, util;

def load_mnist_train(train_amount):
    data = datasets.MNIST("../data", train=True, download = True)
    imgs, labels = data.train_data.numpy() , data.train_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_mnist_train: return following shape arrays");
    print("imgs:", return_data[0:train_amount].shape);
    print("labels:", labels[0:train_amount].shape);
    return return_data[0:train_amount]/255.0, labels[0:train_amount];

def load_cifar10_train(train_amount):
    data = datasets.CIFAR10("../data", train = True, download = True);
    imgs, labels = data.data[0:train_amount], np.array(data.targets[0:train_amount]);
    print("load_cifar_train: return following shape arrays");
    print("imgs:", imgs.shape);
    print("labels:", labels.shape);
    plot.imview(util.tiledict(imgs.transpose(1,2,3,0)[:, :, :, :25]));
    return np.float64(imgs)/255.0, labels;

def load_cifar10_test(test_amount):
    data = datasets.CIFAR10("../data", download = True);
    imgs, labels = data.data, np.array(data.targets);
    imgs, labels = imgs[imgs.shape[0]-test_amount:], labels[labels.shape[0]-test_amount]
    print("load_cifar10_test: return following shape arrays");
    print("imgs:", imgs.shape);
    print("labels:", labels.shape);
    plot.imview(util.tiledict(imgs.transpose(1,2,3,0)[:, :, :, :25]));
    return np.float64(imgs)/255.0, labels;
    
trimg, trlabel = load_cifar10_train(5000)
teimg, telabel = load_cifar10_test(1000)