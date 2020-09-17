import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# loss1 = np.array([0.552828872, 0.240356225, 0.194670086, 0.174720124, 0.159991776, 0.144247817, 0.131543415, 0.12598604, 0.1171949, 0.113940521, 0.111434668, 0.105700386, 0.103998404, 0.100622092, 0.095217411, 0.09254765, 0.091853959, 0.088973453, 0.086288145, 0.085697018, 0.081565329, 0.081251836, 0.077560432, 0.078428429, 0.074416133, 0.075781926, 0.075437516, 0.072117725, 0.072523807, 0.069723618, 0.069974448, 0.068818232, 0.069446511, 0.065838996, 0.067897472, 0.065853196, 0.063842298, 0.06344096, 0.064179388, 0.061891667, 0.062871728, 0.061800771, 0.063421585, 0.059250774, 0.060449895, 0.058704296, 0.058619842, 0.058879062, 0.059490172, 0.058660182, 0.05512337, 0.055313729, 0.055800745, 0.055870661, 0.053810269, 0.053726402, 0.054368721, 0.053826082, 0.053251847, 0.052637619, 0.052565072, 0.051154363, 0.051411046, 0.050795614, 0.050543117, 0.050343625, 0.049940669, 0.051106484, 0.050529716, 0.05030451, 0.048987102, 0.048427232, 0.047392219, 0.046531599, 0.047939342, 0.047864573, 0.047842218, 0.047505565, 0.04726303, 0.045835573, 0.046375986, 0.045389399, 0.046618062, 0.044284059, 0.046274792, 0.044752407, 0.044269525, 0.045368918, 0.043984986, 0.043337595, 0.043174268, 0.043115512, 0.042339096, 0.043922929, 0.042673556, 0.042474595, 0.042061117, 0.042465483, 0.042446229, 0.04208593, 0.041322517, 0.041070007, 0.042226392, 0.0404531, 0.04023483, 0.040830341, 0.040868031, 0.039193244, 0.040948412, 0.040439979, 0.039796663, 0.039712284, 0.038726738, 0.03974584, 0.039291298, 0.039467622, 0.037859642, 0.038161404, 0.038679588, 0.039163079, 0.038583076, 0.038952369, 0.037709098, 0.03764388, 0.037921254, 0.036873654, 0.03738702, 0.038870158, 0.037275421, 0.036988105, 0.037393462, 0.037675715, 0.036993204, 0.036895828, 0.036401066, 0.036447794, 0.036484525, 0.037468086, 0.03568978, 0.036178539, 0.036352319, 0.036774255, 0.035309313, 0.035110899, 0.035386268, 0.034941489, 0.035332058, 0.034790668, 0.035631591, 0.035265116])
# loss2 = np.array([0.542828872, 0.230356225, 0.184670086, 0.164720124, 0.139991776, 0.124247817, 0.121543415, 0.12598604, 0.1071949, 0.153940521, 0.111434668, 0.095700386, 0.103998404, 0.100622092, 0.095217411, 0.09254765, 0.091853959, 0.088973453, 0.086288145, 0.085697018, 0.081565329, 0.081251836, 0.077560432, 0.078428429, 0.074416133, 0.075781926, 0.075437516, 0.072117725, 0.072523807, 0.069723618, 0.069974448, 0.068818232, 0.069446511, 0.065838996, 0.067897472, 0.065853196, 0.063842298, 0.06344096, 0.064179388, 0.061891667, 0.062871728, 0.061800771, 0.063421585, 0.059250774, 0.060449895, 0.058704296, 0.058619842, 0.058879062, 0.059490172, 0.058660182, 0.05512337, 0.055313729, 0.055800745, 0.055870661, 0.053810269, 0.053726402, 0.054368721, 0.053826082, 0.053251847, 0.052637619, 0.052565072, 0.051154363, 0.051411046, 0.050795614, 0.050543117, 0.050343625, 0.049940669, 0.051106484, 0.050529716, 0.05030451, 0.048987102, 0.048427232, 0.047392219, 0.046531599, 0.047939342, 0.047864573, 0.047842218, 0.047505565, 0.04726303, 0.045835573, 0.046375986, 0.045389399, 0.046618062, 0.044284059, 0.046274792, 0.044752407, 0.044269525, 0.045368918, 0.043984986, 0.043337595, 0.043174268, 0.043115512, 0.042339096, 0.043922929, 0.042673556, 0.042474595, 0.042061117, 0.042465483, 0.042446229, 0.04208593, 0.041322517, 0.041070007, 0.042226392, 0.0404531, 0.04023483, 0.040830341, 0.040868031, 0.039193244, 0.040948412, 0.040439979, 0.039796663, 0.039712284, 0.038726738, 0.03974584, 0.039291298, 0.039467622, 0.037859642, 0.038161404, 0.038679588, 0.039163079, 0.038583076, 0.038952369, 0.037709098, 0.03764388, 0.037921254, 0.036873654, 0.03738702, 0.038870158, 0.037275421, 0.036988105, 0.337393462, 0.237675715, 0.136993204, 0.136895828, 0.036401066, 0.036447794, 0.036484525, 0.097468086, 0.03568978, 0.636178539, 0.236352319, 0.136774255, 0.025309313])

# sns.distplot(loss1, bins=50, kde=True);
# sns.distplot(loss2, bins=50, kde=True);
# plt.show()
#
#
# from scipy.stats import ks_2samp
#
# statistic, p_value = ks_2samp(loss1, loss2)
# print("statistic, p_value:", statistic, p_value)
#
# n_samples, n_features = 100, 12
# rng = np.random.RandomState(50)
# X = rng.randn(n_samples, n_features)
#
# from pyts.image import RecurrencePlot
# rp = RecurrencePlot(dimension=1, percentage=10)
# X_rp = rp.fit_transform(X)
#
# plt.figure(figsize=(8,8))
# plt.imshow(X_rp[0])
# plt.show()
#
# import torch
# from torch import nn
#
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)
# loss = nn.CrossEntropy(input, target)
# loss.backward()


# df = pd.read_csv('result/8622146_result_comparison.csv', delimiter=',')
# df.head()
# df['method'].unique()
# sns.boxplot(x='method', y='value', data=df)
# plt.show()
#
# df2 = pd.read_csv('result/8622146_decision_comparison.csv', delimiter=',')
#
#
# def plot_decision(normal, decision, method, ax):
#     ax.plot(normal, label='minutely', color='gray')
#     ax.plot(decision, label=method, color='red')
#     ax.set_title(f'decision of {method}')
#     ax.legend()
#
#
# fig, axs = plt.subplots(
#     nrows=5,
#     ncols=1,
#     sharex=True,
#     sharey=True,
#     figsize=(8, 20)
# )
#
# for i, data in enumerate(df2.head()):
#     normal = df2['pH']
#     if data == 'act_time' or data == 'pH':
#         continue
#     i = i - 2
#     decision = df2[data]
#     method = data
#     plot_decision(normal, decision, method, ax=axs[i])
#
#
# fig.tight_layout()
#
#
# print("the end")

import time
import numpy as np
from AutoEncoderBasedEvaluation import LstmAutoEncoderModel
from DensityBasedEvaluation import DensityModel
from DistanceBasedEvaluation import DTWModel
from VariationalAutoEncoderBasedEvaluation import CnnVariationalAutoEncoderModel


# temporary
# windowSize = 12
# maxEpoch = 150
# paramIndex = 3
# learningRate = 1e-3
# threshold = 0.05
# embeddingDim = 128
#
# model = LstmAutoEncoderModel(windowSize, maxEpoch, paramIndex, learningRate, threshold)
# train, valid, lengthOfSubsequence, numberOfFeatures = model.preProcess()
# autoEncoder = model.train(train, valid, lengthOfSubsequence, numberOfFeatures)
# model.setThreshold(autoEncoder, train, valid)
# model.saveModel(autoEncoder)


# # temporary
# model = LstmAutoEncoderModel(windowSize, 0, paramIndex, 0, threshold)
# autoEncoder = model.loadModel()
# train, valid, lengthOfSubsequence, numberOfFeatures = model.preProcess()
# model.setThreshold(autoEncoder, train, valid)
# model.saveModel(autoEncoder)
#
# model = CnnVariationalAutoEncoderModel(windowSize, maxEpoch, paramIndex, learningRate, threshold)
# variationalAutoEncoder = model.loadModel()
# train, valid = model.preProcess()
# model.setThreshold(variationalAutoEncoder, train, valid)
# model.saveModel(variationalAutoEncoder)
