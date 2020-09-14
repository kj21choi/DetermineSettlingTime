import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fastdtw import fastdtw

import DataLoader
from Model import Model


class DTWModel(Model):
    def __init__(self, windowSize, paramIndex):
        self.windowSize = windowSize  # 1 = 5 min
        self.maxEpoch = 0
        self.paramIndex = paramIndex
        self.learningRate = 0
        self.threshold = 0
        self.normalData = DataLoader.NormalDataLoader(self.paramIndex, 'train')
        self.unstableData = DataLoader.UnstableDataLoader(self.paramIndex, 'test')

    def preProcess(self):
        print('paramIndex:', self.paramIndex)

        stable = self.normalData.data.x_data
        unstable = self.unstableData.data.x_data

        # plot distribution [optional]
        # sns.distplot(stable, label="train")
        # sns.distplot(unstable[:200], label="test")
        # plt.legend()
        # plt.show()

        return stable, unstable

    def train(self, stable, unstable):
        dtwDistances = []
        for i in range(math.floor(len(stable) / self.windowSize) - 1):
            for j in range(math.floor(len(stable) / self.windowSize) - 1 - (i + 1)):
                j = j + (i + 1)
                [distance, path] = fastdtw(stable[i * self.windowSize: (i + 1) * self.windowSize],
                                           stable[j * self.windowSize: (j + 1) * self.windowSize])
                dtwDistances.append(distance)
            print(f'cur({i}), end({math.floor(len(stable) / self.windowSize) - 1}), percent[{np.round((i / (np.floor(len(stable) / self.windowSize) - 1) * 100), 2)}%], dist({np.round(np.mean(dtwDistances), 2)})')
        pValueThreshold = np.mean(dtwDistances)
        return pValueThreshold

    def getThreshold(self):
        return self.threshold

    def saveModel(self):
        np.save('./model/' + str(self.paramIndex) + '_mean_dtw_distance', self.threshold)

    def loadModel(self):
        threshold = np.load('./model/' + str(self.paramIndex) + '_mean_dtw_distance.npy')
        return threshold

    def evaluate(self, stable, unstable, threshold):
        stableStarted = len(unstable) - self.windowSize
        for i in range(stableStarted):
            dtwDistances = []
            for j in range(math.floor(len(stable) / self.windowSize) - 1):
                distance, path = fastdtw(stable[j * self.windowSize: (j + 1) * self.windowSize]
                                         , unstable[i: i + self.windowSize])
                dtwDistances.append(distance)
            meanDtwDistance = np.mean(dtwDistances)
            if meanDtwDistance < threshold:
                print(f'threshold:{threshold}')
                print(f'meanDtwDistance: {np.round(np.float(meanDtwDistance), 3)}')
                stableStarted = i
                break

        self.printResult(stable, unstable, stableStarted)

    def printResult(self, stable, unstable, stableStarted):
        stableMean, stableStd = float(np.mean(stable)), float(np.std(stable))
        resultMean, resultStd = float(np.mean(unstable)), float(np.std(unstable))
        print('stableMean:', np.round(stableMean, 2), ' vs. resultMean: ', np.round(resultMean, 2))
        print('stableStd: ', np.round(stableStd, 3), ' vs. resultStd: ', np.round(resultStd, 3))
        print("==" * 30)
        print("unstable time:", self.unstableData.data.time_axis['act_time'].get(0))
        print("settling time:", stableStarted * 5, "minutes")
        print("stable time:", self.unstableData.data.time_axis['act_time'].get(stableStarted))
        print("decision time:", self.unstableData.data.time_axis['act_time'].get(stableStarted + self.windowSize))
        print("~~" * 30)
        print()
        return
