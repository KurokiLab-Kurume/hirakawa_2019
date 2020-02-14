import numpy as np;
from sporco import linalg, util, metric, plot;
#%%実験のオプションを設定
def make_option():
    opt = {};
    opt['train_amount'] = 1000;
    opt['test_amount'] = 1000;
    opt['num_remaining'] = [3, 4, 7, 6, 8, 16];
    opt['filter_size'] = [(5, 5, 6), (5, 5, 6, 16)];
    return opt;
#%%


#%%
opt = make_option()
train_data, train_label = load_mnist_train(opt['train_amount'])
test_data, test_label = load_mnist_test(opt['test_amount'])
train_data = train_data.transpose(1,2,0)
test_data = test_data.transpose(1,2,0)

#%%CSC第１層

from original_sporco.dictlrn import cbpdndl;

np.random.seed(12345);
#フィルタのサイズ設定 ---> (x,y,channel)
D00 = np.random.randn(5, 5, 6);

#畳み込み辞書学習の実行
lmbda0 = 0.5;
opt0 = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 400,
                            'CBPDN': {'rho': 50.0*lmbda0 + 0.5, 'NonNegCoef': True},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns');
d0 = cbpdndl.ConvBPDNDictLearn(D00, train_data, lmbda0, opt0, dmethod='cns');
D01 = d0.solve();
print("ConvBPDNDictimport numpy as np;Learn solve time: %.2fs" % d0.timer.elapsed('solve'));

#学習済み辞書と辞書の初期値の比較
fig =  plot.figure(figsize=(20, 10));
ax1 = fig.add_subplot(121);
ax2 = fig.add_subplot(122);
plot.imview(util.tiledict(D00), fig=fig, ax=ax1, title="initial dictionary");
plot.imview(util.tiledict(D01.squeeze()), fig=fig, ax=ax2, title="learned dictionary");

#coef ---> スパースなマップ (x, y, 枚数, channel)
coef0 = np.array(d0.getcoef().squeeze());

#%%
print(np.sum(coef0<0))
#%%入力とフィルタ、及びスパースマップの可視化
fig = plot.figure(figsize=(18, 6));
ax1 = fig.add_subplot(131);
ax2 = fig.add_subplot(132);
ax3 = fig.add_subplot(133);
plot.imview(train_data[:, :, 5], fig=fig, ax=ax1, title="input digit");
plot.imview(util.tiledict(D01.squeeze()), fig=fig, ax=ax2, title="filters");
plot.imview(util.tiledict(coef0[:, :, 5, :]), fig=fig,ax=ax3, title="sparse map(response)");

#%% CSC第２層

from original_sporco.dictlrn import cbpdndl;

np.random.seed(12345);
#フィルタのサイズ設定 ---> (x,y,channel)
D10 = np.random.randn(5, 5, 6, 16);

#畳み込み辞書学習の実行
lmbda1 = 0.025;
opt1 = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                            'CBPDN': {'rho': 50.0*lmbda1 + 0.5, 'NonNegCoef': True},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns');
d1 = cbpdndl.ConvBPDNDictLearn(D10, coef0.transpose(0,1,3,2), lmbda1, opt1, dmethod='cns');
D11 = d1.solve();
print("ConvBPDNDictLearn solve time: %.2fs" % d1.timer.elapsed('solve'));

"""
#学習済み辞書と辞書の初期値の比較
fig =  plot.figure(figsize=(20, 10));
ax1 = fig.add_subplot(121);
ax2 = fig.add_subplot(122);
plot.imview(util.tiledict(D0), fig=fig, ax=ax1, title="initial dictionary");
plot.imview(util.tiledict(D1.squeeze()), fig=fig, ax=ax2, title="learned dictionary");
"""
#coef ---> スパースなマップ (x, y, 枚数, channel)
coef1 = np.array(d1.getcoef().squeeze());

#%%
print(np.sum(coef1>0));
#%%
feature1 = coef1.transpose(0, 1, 3, 2).reshape((32*32*16, -1));
plot_3d(feature1, train_label);
#%%


#%%
from original_sporco.admm import cbpdn

test_opt0 = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                              'RelStopTol': 5e-3, 'AuxVarObj': False, 'NonNegCoef':True});
b = cbpdn.ConvBPDN(D01, test_data, lmbda0, test_opt0, dimK=0);
test_coef0 = b.solve();
print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'));
#%%
from original_sporco.admm import cbpdn

test_opt1 = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                              'RelStopTol': 5e-3, 'AuxVarObj': False, 'NonNegCoef':True});
b = cbpdn.ConvBPDN(D11.squeeze(), test_coef0.transpose(0,1,4,3,2).squeeze(), lmbda1, test_opt1);
test_coef1 = b.solve();
print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'));

#%%
test_feature = test_coef1.transpose(0, 1, 4, 2, 3).reshape((32*32*16, -1))
