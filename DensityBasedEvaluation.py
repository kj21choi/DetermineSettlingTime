import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

import DataLoader
from Model import Model


class DensityModel(Model):
    def __init__(self, windowSize, paramIndex, threshold):
        self.windowSize = windowSize  # 1 = 5 min
        self.maxEpoch = 0
        self.paramIndex = paramIndex
        self.learningRate = 0
        self.threshold = threshold
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

    def getThreshold(self):
        pValueThreshold = 0.05
        return pValueThreshold

    def evaluate(self, stable, unstable, threshold):
        stableStarted = len(unstable) - self.windowSize

        # t-test(mean) + ansari-bradley(var): pH 결과는 좋지만 current 결과 나쁨(중심치 변동)
        for i in range(stableStarted):
            isStable = self.compareMeanVariance(stable, unstable[i: i + self.windowSize], threshold)
            if isStable:
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
        print("decision time:", self.unstableData.data.time_axis['act_time'].get(stableStarted + self.windowSize - 1))
        print("~~" * 30)
        print()
        return

    @staticmethod
    def compareMeanVariance(stable, unstable, threshold):
        statsMean, PvalueMean = scipy.stats.ttest_ind(stable, unstable, equal_var=True)
        statsVariance, pValueVariance = scipy.stats.ansari(stable, unstable)
        if PvalueMean >= threshold and pValueVariance >= threshold:
            print(f'threshold:{threshold}')
            print(f'p_value of T-test: {np.round(np.float(PvalueMean), 3)}')
            print(f'p_value of AB-Test:{np.round(pValueVariance, 3)}')
            return True
        else:
            return False


# Jensen-Shannon Divergence:
# method to compute the Jensen Distance between two probability distributions
def js_divergence(p, q):
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


# Kullback–Leibler Divergence:
# method to compute the Jensen Distance between two probability distributions
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

