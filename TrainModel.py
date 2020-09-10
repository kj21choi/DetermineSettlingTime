from sklearn.preprocessing import MinMaxScaler

import DataLoader
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import DataLoader
import torch
from torch import nn
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

from DensityBasedEvaluation import DensityBasedModel
from DistanceBasedEvaluation import DTWBasedModel
from LSTMmodel import RecurrentAutoEncoder
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import torch.nn.functional as F

import DataLoader
import torch
from torch import nn
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

from VAEmodel import VAE, VAE2
from pyts.image import RecurrencePlot

import time

"""
1. Set hyper-parameters
2. Select model
3. Train by model
    - Data pre-process
    - Train
    - Validate
    - Save Model 
"""
modelTypes = ['Density', 'DTW', 'LSTM-AE', 'CNN-VAE']  # 0, 1, 2, 3
selectedModel = modelTypes[1]
windowSize = 12
maxEpoch = 150
paramIndex = 8726725
learningRate = 1e3
threshold = 0.05


start = time.time()  # start time of training
if selectedModel == 'Density':
    model = DensityBasedModel(windowSize=windowSize, paramIndex=paramIndex, threshold=threshold)
    # training is unnecessary

elif selectedModel == 'DTW':
    model = DTWBasedModel(windowSize=windowSize, paramIndex=paramIndex)
    stable, unstable = model.preProcess()
    threshold = model.train(stable, unstable)
    model.setThreshold(threshold)
    model.saveModel()

elif selectedModel == 'LSTM-AE':
    model = DensityBasedModel()
    model.preProcess()

elif selectedModel == 'CNN-VAE':
    model = DensityBasedModel()
    model.preProcess()


end = time.time()  # end time of training
print("Training time:", (end - start) / 60, "minutes.")