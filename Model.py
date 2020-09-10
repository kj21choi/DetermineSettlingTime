class Model:

    def __init__(self):
        # Hyper parameter
        self.windowSize = 12  # 1 = 5 min
        self.maxEpoch = 150
        self.paramIndex = 0
        self.learningRate = 1e3
        self.threshold = 0.05

    def __init__(self, windowSize, maxEpoch, paramIndex, learningRate, threshold):
        # Hyper parameter
        self.windowSize = windowSize  # 1 = 5 min
        self.maxEpoch = maxEpoch
        self.paramIndex = paramIndex
        self.learningRate = learningRate
        self.threshold = threshold

    def preProcess(self):
        return

    def train(self):
        return

    def validate(self):
        return

    def setThreshold(self, threshold):
        self.threshold = threshold

    def getThreshold(self):
        return

    def saveModel(self):
        return

    def loadModel(self):
        return

    def evaluate(self):
        return
