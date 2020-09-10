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


"""
1. Select model
2. Evaluate model
    - Data pre-process
    - Load model
    - Set Threshold
    - Evaluate settling time
"""
modelTypes = ['Density', 'DTW', 'LSTM-AE', 'CNN-VAE']  # 0, 1, 2, 3
selectedModel = modelTypes[1]

windowSize = 12
paramIndex = 8726725
threshold = 0.05

if selectedModel == 'Density':
    model = DensityBasedModel(windowSize=windowSize, paramIndex=paramIndex, threshold=threshold)
    stable, unstable = model.preProcess()
    threshold = model.getThreshold()
    model.evaluate(stable, unstable, threshold)
    # Load model is unnecessary

elif selectedModel == 'DTW':
    model = DTWBasedModel(windowSize=windowSize, paramIndex=paramIndex)
    stable, unstable = model.preProcess()
    threshold = model.loadModel()
    model.setThreshold(threshold)
    model.evaluate(stable, unstable, threshold)

elif selectedModel == 'LSTM-AE':
    model = DensityBasedModel()
    model.preProcess()

elif selectedModel == 'CNN-VAE':
    model = DensityBasedModel()
    model.preProcess()