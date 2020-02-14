#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:02:11 2019

@author: tom
"""

from sklearn.decomposition import NMF;
from scipy import sparse;
from sklearn.preprocessing import normalize;
import numpy as np;

def cr_sm(coef, label, n_components=16):
    dic = {};
    
    return ;

def squeeze_class(coef, label, i):
    return coef[label == i];

def conduct_nmf(coef, num_remaining):
    nmf = NMF(n_components=num_remaining);
    nmf.fit(coef);
    return nmf;

def restore_from_nmf(data, nmf):
    return nmf.transfrom(nmf.inverse_transform(data));

def take_similarity(d1, d2):
    d1 = normalize(d1);
    d2 = normalize(d2);
    cor = [];
    for i in range(d1.shape[0]):
        cor.append(np.dot(d1[i], d2[i]));
    return np.array(cor);

    