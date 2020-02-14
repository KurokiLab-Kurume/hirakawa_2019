#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:56:36 2019

@author: kuroki-lab
"""
from sklearn.decomposition import NMF
import numpy as np;
import scipy as sp;

def create_cone_NMF(feature, num):
    nmf = NMF(n_components=num);
    W = nmf.fit_transform(feature);
    H = nmf.components_;
    return nmf, H, W;

def create_cone_strict(feature, num):
    return 0;

def create_cone_PCA(feature, num):
    return 0;

def create_cone_circle(feature, num):
    return 0;
#%%
'''
# feature: (データ数, 特徴)
feature = np.random.uniform(0.0, 1.0, (500, 1024));
nmf = NMF(n_components=8);
W = nmf.fit_transform(feature)
H = nmf.components_
'''