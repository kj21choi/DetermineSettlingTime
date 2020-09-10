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

"""
1. Set hyper-parameters
2. Select model
3. Train by model
    - Data pre-process
    - Train
    - Validate
    - Save Model 
"""
modelTypes = ['Density', 'NN-DTW', 'LSTM-AE', 'CNN-VAE']  # 0, 1, 2, 3
selectedModel = modelTypes[0]

if selectedModel == 'Density':
    model = DensityBasedModel()
    # training is unnecessary

elif selectedModel == 'NN-DTW':
    model = DensityBasedModel()
    model.preProcess()

elif selectedModel == 'LSTM-AE':
    model = DensityBasedModel()
    model.preProcess()

elif selectedModel == 'CNN-VAE':
    model = DensityBasedModel()
    model.preProcess()
