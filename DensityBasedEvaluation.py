from sklearn.preprocessing import MinMaxScaler

import DataLoader
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from TrainModel import Model


class DensityBasedModel(Model):
    def __init__(self):
        self.normalData = DataLoader.NormalDataLoader(self.paramIndex)
        self.unstableData = DataLoader.UnstableDataLoader(self.paramIndex)

    def preProcess(self):
        print('paramIndex:', self.paramIndex)

        stable = self.normalData.data.x_data
        unstable = self.unstableData.data.x_data

        # plot distribution
        sns.distplot(stable, label="train")
        sns.distplot(unstable[:200], label="test")
        plt.legend()
        plt.show()

        return stable, unstable

    def getThreshold(self):
        pValueThreshold = 0.05
        return pValueThreshold

    def evaluate(self, stable, unstable, threshold):
        stableStarted = len(unstable) - self.windowSize
        isStable = False

        # t-test(mean) + ansari-bradley(var): pH 결과는 좋지만 current 결과 나쁨(중심치 변동)
        for i in range(len(unstable) - self.windowSize):
            print("index: ", i)
            isStable = self.compareMeanVariance(stable, unstable[i: i + self.windowSize], i, threshold)
            if isStable:
                stableStarted = i
                break

        self.printResult(stable, unstable, stableStarted)

    def printResult(self, stable, unstable, stableStarted):
        stableMean = np.mean(stable)
        stableStd = np.std(stable)
        resultMean = np.mean(unstable)
        resultStd = np.std(unstable)
        print(f'stableMean: {np.round(stableMean, 2)} vs. resultMean {np.round(resultMean, 2)}')
        print(f'stableStd: {np.round(stableStd, 3)} vs. resultStd {np.round(resultStd, 3)}')
        print("==" * 30)
        print("unstable time:", self.unstableData.data.time_axis['act_time'].get(0))
        print("stable time:", self.unstableData.data.time_axis['act_time'].get(stableStarted))
        print("decision time:", self.unstableData.data.time_axis['act_time'].get(stableStarted + self.windowSize - 1))
        return

    @staticmethod
    def compareMeanVariance(stable, unstable, threshold):
        statsMean, PvalueMean = scipy.stats.ttest_ind(stable, unstable, equal_var=True)
        statsVariance, pValueVariance = scipy.stats.ansari(stable, unstable)
        if PvalueMean >= threshold and pValueVariance >= threshold:
            print(f'p_value of T-test: {np.round(PvalueMean, 3)}, p_value of AB-Test:{np.round(pValueVariance, 3)}')
            print(f'threshold:{threshold}')
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

