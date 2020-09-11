from DensityBasedEvaluation import DensityBasedModel
from DistanceBasedEvaluation import DTWBasedModel

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