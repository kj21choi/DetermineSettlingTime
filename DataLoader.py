import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import TimeSeries


class DataLoader:

    def __init__(self, paramIndex, seriesType):
        self.data = TimeSeries.TimeSeries(paramIndex, seriesType)
        self.numberOfBlocks = 10

    # Save plot
    def saveFigure(self, index):
        figure, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 6))
        sns.lineplot(data=self.data.x_data, ax=axes[0, 0])
        sns.distplot(self.data.x_data, ax=axes[1, 0])
        plt.setp(axes, yticks=[])
        plt.tight_layout()
        plt.savefig(index + '_train.png')  # plt.show()

    # Plot time series & distribution
    def plotFigure(self):
        figure, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 6))
        sns.lineplot(data=self.data.x_data, ax=axes[0, 0])
        sns.distplot(self.data.x_data, ax=axes[1, 0])
        plt.setp(axes, yticks=[])
        plt.tight_layout()
        plt.show()


class NormalDataLoader(DataLoader):
    """ Normal data loader """

    def __init__(self, paramIndex, seriesType):
        self.data = TimeSeries.NormalTimeSeries(paramIndex, seriesType)
        self.numberOfBlocks = 10

    def divideData(self, data, wantToShuffle):
        # Divide data for train:valid= 8:2
        dataToArray = data.to_numpy().flatten()
        lengthOfBlock = int(len(dataToArray) / self.numberOfBlocks)  # (299 / 10) = 29
        dataSegment = np.zeros((self.numberOfBlocks, lengthOfBlock), dtype=np.float)
        for i in range(self.numberOfBlocks - 1):
            dataSegment[i, :] = dataToArray[i * lengthOfBlock: (i + 1) * lengthOfBlock]

        # rest = lengthOfBlock - (lengthOfBlock * numberOfBlocks - data.len)
        dataSegment[i + 1, :] = dataToArray[(i + 1) * lengthOfBlock:(i + 2) * lengthOfBlock]

        # Shuffle
        if wantToShuffle:
            np.random.shuffle(dataSegment)

        # divide
        trainData = dataSegment[2:10, :].reshape(-1, 1)
        validData = dataSegment[:2, :].reshape(-1, 1)

        return trainData, validData


class UnstableDataLoader:
    """ Unstable data loader """

    def __init__(self, paramIndex, seriesType):
        self.data = TimeSeries.UnstableTimeSeries(paramIndex, seriesType)
