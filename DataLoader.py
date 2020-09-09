import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import TimeSeries


class DataLoader:

    def __init__(self, paramIndex):
        self.data = TimeSeries.TimeSeries(paramIndex)
        self.numberOfBlocks = 10


    def divide_data(self, data, is_train):
        # Divide data for train(8), test(2)
        data_numpy = data.x_data.to_numpy().flatten()
        blockLength = int(data.len / self.numberOfBlocks)  # (299 / 10) = 29
        data_seg = np.zeros((self.numberOfBlocks, blockLength), dtype=np.float)
        for i in range(self.numberOfBlocks - 1):
            data_seg[i, :] = data_numpy[i * blockLength: (i + 1) * blockLength]

        # restLength = blockLength - (blockLength * BLOCK_NUM - data.len)
        data_seg[i + 1, :] = data_numpy[(i + 1) * blockLength:(i + 2) * blockLength]

        # Shuffle
        if is_train:
            np.random.shuffle(data_seg)

        # divide
        train_data = data_seg[0:8, :].reshape(-1, 1)
        test_data = data_seg[8:, :].reshape(-1, 1)

        return train_data, test_data

    # Plot time series & distribution
    def save_figure(self, index):
        figure, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 6))
        sns.lineplot(data=self.data.x_data, ax=axes[0, 0])
        sns.distplot(self.data.x_data, ax=axes[1, 0])
        plt.setp(axes, yticks=[])
        plt.tight_layout()
        plt.savefig(index + '_train.png')  # plt.show()

    def plot_figure(self):
        figure, axes = plt.subplots(2, 1, squeeze=False, figsize=(10, 6))
        sns.lineplot(data=self.data.x_data, ax=axes[0, 0])
        sns.distplot(self.data.x_data, ax=axes[1, 0])
        plt.setp(axes, yticks=[])
        plt.tight_layout()
        plt.show()


class NormalDataLoader(DataLoader):
    """ Normal data loader """

    def __init__(self, paramIndex):
        self.data = TimeSeries.NormalTimeSeries(paramIndex)


class UnstableDataLoader:
    """ Unstable data loader """

    def __init__(self, paramIndex):
        self.data = TimeSeries.UnstableTimeSeries(paramIndex)
