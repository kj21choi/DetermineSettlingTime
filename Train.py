import time
import numpy as np
from AutoEncoderBasedEvaluation import LstmAutoEncoderBasedModel
from DensityBasedEvaluation import DensityBasedModel
from DistanceBasedEvaluation import DTWBasedModel

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
selectedModel = modelTypes[2]
windowSize = 12
maxEpoch = 150
paramIndex = 8726725
learningRate = 1e-3
threshold = 0.05
embeddingDim = 128

start = time.time()  # start time of training

if selectedModel == 'Density':
    model = DensityBasedModel(windowSize=windowSize, paramIndex=paramIndex, threshold=threshold)
    # training is not necessary

elif selectedModel == 'DTW':
    model = DTWBasedModel(windowSize=windowSize, paramIndex=paramIndex)
    stable, unstable = model.preProcess()
    threshold = model.train(stable, unstable)
    model.setThreshold(threshold)
    model.saveModel()

elif selectedModel == 'LSTM-AE':
    model = LstmAutoEncoderBasedModel(windowSize, maxEpoch, paramIndex, learningRate, threshold)
    train, valid, lengthOfSubsequence, numberOfFeatures = model.preProcess()
    autoEncoder = model.train(train, valid, lengthOfSubsequence, numberOfFeatures)
    model.setThreshold(autoEncoder, train, valid)
    model.saveModel(autoEncoder)

elif selectedModel == 'CNN-VAE':
    model = DensityBasedModel()
    model.preProcess()

end = time.time()  # end time of training
print("Training time:", np.round((end - start) / 60, 0), "minutes.")