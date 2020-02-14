#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:31:33 2019

@author: kuroki-lab
"""

from sporco import util;
import numpy as np;
#----------------------------------------------------
# Parameters
#   input_data : (データ数,チャンネル,高さ,横幅)の4次元配列からなる入力データ
#   filter_h : フィルターの高さ
#   filter_w : フィルターの横幅
#   stride : ストライド
#   pad : パディング
# Returns
#   col : 2次元配列 
#----------------------------------------------------

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def forward(x, pool_h, pool_w, stride=1, pad = 0):
        N, C, H, W = x.shape
        out_h = int(1 + (H - pool_h) / stride)
        out_w = int(1 + (W - pool_w) / stride)

        col = im2col(x, pool_h, pool_w, stride, pad)
        col = col.reshape(-1, pool_h*pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C)
        
        x = x
        arg_max = arg_max
        return out
    
def back(x,stride):
    h, w = x.shape[0]*2, x.shape[1]*2;
    array = np.zeros((h, w));
    