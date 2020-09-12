from AutoEncoderBasedEvaluation import LstmAutoEncoderModel
from DensityBasedEvaluation import DensityModel
from DistanceBasedEvaluation import DTWModel
from VariationalAutoEncoderBasedEvaluation import CnnVariationalAutoEncoderModel

"""
1. Select model
2. Evaluate model
    - Data pre-process
    - Load model
    - Set Threshold
    - Evaluate settling time
"""
modelTypes = ['Density', 'DTW', 'LSTM-AE', 'CNN-VAE']  # 0, 1, 2, 3
selectedModel = modelTypes[3]
windowSize = 12
paramIndex = 8726725
threshold = 0.05
print('selected model: ', selectedModel)

if selectedModel == 'Density':
    model = DensityModel(windowSize=windowSize, paramIndex=paramIndex, threshold=threshold)
    stable, unstable = model.preProcess()
    threshold = model.getThreshold()
    model.evaluate(stable, unstable, threshold)
    # Load model is unnecessary

elif selectedModel == 'DTW':
    model = DTWModel(windowSize=windowSize, paramIndex=paramIndex)
    stable, unstable = model.preProcess()
    threshold = model.loadModel()
    model.setThreshold(threshold)
    model.evaluate(stable, unstable, threshold)

elif selectedModel == 'LSTM-AE':
    model = LstmAutoEncoderModel(windowSize, 0, paramIndex, 0, threshold)
    autoEncoder = model.loadModel()
    model.evaluate(autoEncoder)

elif selectedModel == 'CNN-VAE':
    model = CnnVariationalAutoEncoderModel(windowSize, 0, paramIndex, 0, threshold)
    variationalAutoEncoder = model.loadModel()
    model.evaluate(variationalAutoEncoder)
