import time
import numpy as np
from AutoEncoderBasedEvaluation import LstmAutoEncoderModel
from DensityBasedEvaluation import DensityModel
from DistanceBasedEvaluation import DTWModel
from VariationalAutoEncoderBasedEvaluation import CnnVariationalAutoEncoderModel

"""
1. Set hyper-parameters
2. Select model
3. Train by model
    - Data pre-process
    - Train
    - Validate
    - Save Model 
"""
windowSize = 12
maxEpoch = 150
paramIndex = 5
learningRate = 1e-3
threshold = 0.05
embeddingDim = 128

modelTypes = ['Density', 'DTW', 'LSTM-AE', 'CNN-VAE']  # 0, 1, 2, 3
for i, _ in enumerate(modelTypes):
    selectedModel = modelTypes[i]
    print('selected model: ', selectedModel)
    start = time.time()  # start time of training

    if selectedModel == 'Density':
        model = DensityModel(windowSize=windowSize, paramIndex=paramIndex, threshold=threshold)
        # training is not necessary

    elif selectedModel == 'DTW':
        model = DTWModel(windowSize=windowSize, paramIndex=paramIndex)
        stable, unstable = model.preProcess()
        threshold = model.train(stable, unstable)
        model.setThreshold(threshold)
        model.saveModel()

    elif selectedModel == 'LSTM-AE':
        model = LstmAutoEncoderModel(windowSize, maxEpoch, paramIndex, learningRate, threshold)
        train, valid, lengthOfSubsequence, numberOfFeatures = model.preProcess()
        autoEncoder = model.train(train, valid, lengthOfSubsequence, numberOfFeatures)
        model.setThreshold(autoEncoder, train, valid)
        model.saveModel(autoEncoder)

    elif selectedModel == 'CNN-VAE':
        model = CnnVariationalAutoEncoderModel(windowSize, maxEpoch, paramIndex, learningRate, threshold)
        train, valid = model.preProcess()
        variationalAutoEncoder = model.train(train, valid)
        model.setThreshold(variationalAutoEncoder, train, valid)
        model.saveModel(variationalAutoEncoder)

    end = time.time()  # end time of training
    print("Training time:", np.round((end - start) / 60, 0), "minutes.")