#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:26:19 2019

@author: tom
"""
import numpy as np;
from load_dataset import *;
from sporco.admm import cbpdn;
import csv;
import sys;
from conduct_csc import csc;
from sklearn.decomposition import NMF;
from sklearn.decomposition import PCA;
from sklearn.preprocessing import normalize;
from sklearn.metrics.pairwise import cosine_similarity;
from scipy.optimize import nnls;
from sklearn.neural_network import MLPClassifier
from sklearn import svm;

def train_svm(feature, label):
    clf = svm.SVC(C=5, gamma=0.05);
    clf.fit(feature, label);
    train_result = clf.predict(feature);
    precision = sum(train_result == label)/label.shape[0];
    print('Train precision: ', precision);
    return clf;

def test_svm(clf, feature, label):
    test_result = clf.predict(feature);
    precision = sum(test_result == label)/label.shape[0];
    print('Test precision: ', precision);
    return precision;

def SVM_Classifier(train_feature, train_label, test_feature, test_label):
    clf = train_svm(train_feature, train_label);
    return test_svm(clf, test_feature, test_label);

def train_nn(feature, label):
    clf = MLPClassifier(solver= 'adam', alpha=1e-5, hidden_layer_sizes=(128, ), random_state=1, max_iter=1000)
    clf.fit(feature, label);
    train_result = clf.predict(feature);
    precision = sum(train_result == label)/label.shape[0];
    print('Train precision: ', precision);
    return clf;

def test_nn(clf, feature, label):
    test_result = clf.predict(feature);
    precision = sum(test_result == label)/label.shape[0];
    print('Test precision: ', precision);
    return precision;

def NN_lLassifier(train_feature, train_label, test_feature, test_label):
    nn = train_nn(train_feature, train_label);
    return test_nn(nn, test_feature, test_label);
# 錘を作るための関数郡

def create_cone_NMF(feature, num):
    nmf = NMF(n_components=num);
    W = nmf.fit_transform(feature);
    H = nmf.components_;
    return nmf, H, W;

def restore_from_cone_NMF(feature, nmf):
    restore = nmf.inverse_transform(nmf.transform(feature));
    return restore;

def NMF_Classifier(train_feature, train_label, test_feature, test_label, from_, to_):
    forcsv = []
    forcsv.append(["EXPERIMENT", "NMF CONE"]);
    forcsv.append(["dim", "accuracy"]);
    for h in range(from_, to_+1):
        NMFs = [];
        Hs = [];
        Ws = [];
        for i in range(10):
            #print(str(i)+"番目のクラスの錐を作成中...");
            nmf, H, W = create_cone_NMF(train_feature[train_label == i], h);
            NMFs.append(nmf);
            Hs.append(H);
            Ws.append(W);
        restores = [];
        for i in range(10):
            #print(str(i)+"番目のクラスの錐で入力を再現中...")
            temp = restore_from_cone_NMF(test_feature, NMFs[i]);
            restores.append(temp);
        CosArray = [];
        for i in range(10):
            cos_array = cosine_similarity(test_feature, restores[i]);
            CosArray.append(np.diag(cos_array));            
        CosArray = np.array(CosArray);
        predict_label = np.argmax(CosArray, axis = 0);
        precision = sum(predict_label == test_label)/test_label.shape[0];
        forcsv.append([h, precision])
        print("d_num="+str(h)+": Test precision: ", precision);
    return forcsv;

def create_cone_PCA(feature, param, ratio):
    feature = normalize(feature);
    pca = PCA(n_components=ratio);
    pca.fit(feature);
    variance = pca.explained_variance_;
    mean = pca.mean_;
    basis = pca.components_;
    cone_basis = [];
    cnt = 0;
    for i in variance:
        cone_basis.append(mean+param*np.sqrt(i)*basis[cnt]);
        cone_basis.append(mean-param*np.sqrt(i)*basis[cnt]);
        cnt=cnt+1;
    cone_basis = np.array(cone_basis);
    return cone_basis;

def restore_from_cone_PCA(cone_basis, feature):
    feature = list(feature);
    restore = [];
    for i in feature:
        x = np.array(nnls(cone_basis.T, i)[0]);
        restore.append(cone_basis.T.dot(x));
    restore = np.array(restore);
    return restore;

def PCA_Classifier(train_feature, train_label, test_feature, test_label, params, ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999]):
    forcsv=[]
    forcsv.append(["EXPERIMENT", "COMPREHENSIVE CONE"])
    for param in params:
        forcsv.append(["scaling param", str(param)])
        forcsv.append(["contribution rate", "accuracy"])
        print("the parameter of scaling : "+str(param))
        for ratio in ratios:
            Cones = [];

            for i in range(10):
                #print(str(i)+"番目のクラスの錐を作成中...");
                cone = create_cone_PCA(train_feature[train_label == i], param, ratio);
                Cones.append(cone);
            restores = [];
            for i in range(10):
                #print(str(i)+"番目のクラスの錐で入力を再現中...")
                temp = restore_from_cone_PCA(Cones[i], test_feature);
                restores.append(temp);
            CosArray = [];
            for i in range(10):
                cos_array = cosine_similarity(test_feature, restores[i]);
                CosArray.append(np.diag(cos_array));

            CosArray = np.array(CosArray);
            predict_label = np.argmax(CosArray, axis = 0);
            precision = sum(predict_label == test_label)/test_label.shape[0];
            print("ratio="+str(ratio)+": Test precision: ", precision);
            forcsv.append([ratio, precision])
    return forcsv;

def power_spectrum(img):
    return np.abs(np.fft.fftshift(np.fft.fft2(img)));
#%%実験のオプションを設定
def make_option(train_amount):
    opt = {};
    # 訓練画像の枚数
    opt['train_amount'] = np.int(train_amount);
    # テスト画像の枚数
    opt['test_amount'] = 1000;
    # フィルタのサイズ
    opt['d_size'] = (5, 5, 1, 1, 6);
    # スパースの尺度
    opt['lmbda'] = 0.05;
    # 繰り返し回数
    opt['Iter'] = 400;
    # NMFによる錐制約部分空間法における基底の数
    opt['d_num'] = 128;
    # PCAによる包括凸錐による部分空間法における累積寄与率のリスト・パラメータ
    opt["ratios"] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999];
    opt["params"] = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0];
    return opt;

#%%
args = sys.argv;
opt = make_option(args[1]);
if args[2]=="fashion":
    train_data, train_label = load_fashion_train(opt["train_amount"]);
    test_data, test_label = load_fashion_test(opt["test_amount"]);
else:
    train_data, train_label = load_mnist_train(opt["train_amount"]);
    test_data, test_label = load_mnist_test(opt["test_amount"]);

train_data = np.expand_dims(train_data.transpose(1,2,0), 2);
test_data = np.expand_dims(test_data.transpose(1,2,0), 2);
test_data = np.expand_dims(test_data, -1);
#%%
# 学習用データセットで辞書学習
print("### 学習用データセットで畳み込み辞書学習");
print("第１層目:フォワードパスの計算中...");
d0, D0, coef0_ = csc(input_ = np.float32(train_data), d_size = opt['d_size'], lmbda = opt['lmbda'], Iter = opt['Iter'], visualize = False);
coef = np.zeros_like(coef0_)
for i in range(opt['train_amount']):
    for j in range(coef0_.shape[4]):
        coef[:, :, 0, i, j] = power_spectrum(coef0_[:, :, 0, i, j]);
#train_feature = coef0_.transpose(3, 0, 1, 2, 4).squeeze().reshape(opt["train_amount"], -1);
train_feature_p = coef.transpose(3, 0, 1, 2, 4).squeeze().reshape(opt["train_amount"], -1);

#%%
if(opt["train_amount"]<256):
    #nmf = NMF(n_components = opt["train_amount"]);
    nmf_p = NMF(n_components = opt["train_amount"]);
else:
    #nmf = NMF(n_components = 256);
    nmf_p = NMF(n_components = 256);
    
#nmf.fit(train_feature);
#train_feature = nmf.transform(train_feature);

nmf_p.fit(train_feature_p);
train_feature_p = nmf_p.transform(train_feature_p);

#%%
# テスト用データセットの係数算出
print("### テスト用データセットの係数算出 ###");

test_opt0 = cbpdn.ConvBPDN.Options({'Verbose': False, 'MaxMainIter': opt["Iter"], 'RelStopTol': 5e-3, 'AuxVarObj': False});
b0 = cbpdn.ConvBPDN(np.float32(D0), np.float32(test_data), opt["lmbda"], test_opt0);
print("第１層目:フォワードパスの計算中...");
test_coef0_ = b0.solve();

test_coef0 = np.zeros_like(test_coef0_);
for i in range(opt['test_amount']):
    for j in range(test_coef0_.shape[4]):
        test_coef0[:, :, 0, i, j] = power_spectrum(test_coef0_[:, :, 0, i, j])
        
#test_feature = test_coef0_.transpose(3, 0, 1, 2, 4).squeeze().reshape(opt["test_amount"], -1);
#test_feature = nmf.transform(test_feature);
test_feature_p = test_coef0.transpose(3, 0, 1, 2, 4).squeeze().reshape(opt["test_amount"], -1);
test_feature_p = nmf_p.transform(test_feature_p);

#%%
#result_pca = PCA_Classifier(train_feature, train_label, test_feature, test_label, opt["params"], opt["ratios"]);
result_pca_p = PCA_Classifier(train_feature_p, train_label, test_feature_p, test_label, opt["params"], opt["ratios"]);

#result_nmf = NMF_Classifier(train_feature, train_label, test_feature, test_label, 1, opt["d_num"]);
result_nmf_p = NMF_Classifier(train_feature_p, train_label, test_feature_p, test_label, 1, opt["d_num"]);

#result_nn = NN_lLassifier(train_feature, train_label, test_feature, test_label);
result_nn_p = NN_lLassifier(train_feature_p, train_label, test_feature_p, test_label);

#result_svm = SVM_Classifier(train_feature, train_label, test_feature, test_label);
result_svm_p = SVM_Classifier(train_feature_p, train_label, test_feature_p, test_label);

#filename = "./result/["+str(opt["train_amount"])+"]"+args[2]+".csv"
filename_p = "./result/exe2/["+str(opt["train_amount"])+"]"+args[2]+"_p.csv"

'''
with open(filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["SVM", result_svm])
    writer.writerow(["NN", result_nn])
    writer.writerows(result_nmf)
    writer.writerows(result_pca)
''' 
with open(filename_p, "w") as g:
    writer = csv.writer(g)
    writer.writerow(["SVM", result_svm_p])
    writer.writerow(["NN", result_nn_p])
    writer.writerows(result_nmf_p)
    writer.writerows(result_pca_p)




