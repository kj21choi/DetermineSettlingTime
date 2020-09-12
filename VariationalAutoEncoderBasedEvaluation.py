import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pyts.image import RecurrencePlot
from scipy import signal
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from statsmodels import api as sm
from torch import nn

import DataLoader
from CnnVariationalAutoEncoder import VariationalAutoEncoder
from Model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnVariationalAutoEncoderModel(Model):
    def __init__(self, windowSize, maxEpoch, paramIndex, learningRate, threshold):
        self.windowSize = windowSize
        self.maxEpoch = maxEpoch
        self.paramIndex = paramIndex
        self.learningRate = learningRate
        self.threshold = threshold
        self.embeddingDim = 128
        self.normalData = DataLoader.NormalDataLoader(self.paramIndex, 'train')
        self.unstableData = DataLoader.UnstableDataLoader(self.paramIndex, 'test')
        self.wantToShuffle = False
        self.statistics = []

    def preProcess(self):
        print('paramIndex:', self.paramIndex)

        stable = self.normalData.data.x_data

        # plot distribution [optional]
        sns.distplot(stable, label="train")
        plt.legend()
        plt.show()

        # mix max scaler
        minMaxScaler = MinMaxScaler()
        minMaxScaler.fit(stable)

        # divide dataset into train set and validation set
        trainData, validationData = self.normalData.divideData(stable, self.wantToShuffle)

        # remove some data for reshaping
        trainDataMissing = trainData.shape[0] % self.windowSize
        validationDataMissing = validationData.shape[0] % self.windowSize
        trainData = trainData[: -trainDataMissing]
        validationData = validationData[: -validationDataMissing]

        # plot dataset [optional]
        print("data shape:", trainData.shape, validationData.shape)
        plt.plot(trainData)
        plt.plot(validationData)
        plt.show()

        # reshape inputs [timesteps, samples] into subsequence (sliding window)
        trainData = trainData.reshape(-1, self.windowSize)  # 12(window)
        validationData = validationData.reshape(-1, self.windowSize)
        print("data shape:", trainData.shape, validationData.shape)

        # collect mean, std
        meanOfTrainData, stdOfTrainData = self.collectMeanStd(trainData)
        meanOfValidationData, stdOfValidationData = self.collectMeanStd(validationData)
        meanOfTrainData += meanOfValidationData
        stdOfTrainData += stdOfValidationData

        # find cycle of repeated trend
        cycle = self.findCycle(stable)

        # save statistic values [left tail, right tail, right tail(std), cycle]
        self.statistics = [np.percentile(meanOfTrainData, 5),
                           np.percentile(meanOfTrainData, 95),
                           np.percentile(stdOfTrainData, 95),
                           cycle]

        # flatten dataset and min-max normalize
        trainData = minMaxScaler.transform(trainData.reshape(-1, 1))
        validationData = minMaxScaler.transform(validationData.reshape(-1, 1))

        # reshape inputs [timesteps, samples] into subsequence (sliding window)
        trainData = trainData.reshape(-1, self.windowSize)
        validationData = validationData.reshape(-1, self.windowSize)

        trainDataTensor = self.convertToTensor(trainData)
        validationDataTensor = self.convertToTensor(validationData)

        return trainDataTensor, validationDataTensor

    @staticmethod
    def findCycle(sequence):
        normalizedStable = sequence - np.mean(sequence)
        acf = sm.tsa.acf(normalizedStable, nlags=len(normalizedStable), fft=False)  # auto correlation
        peaks, _ = find_peaks(acf, height=0)
        if peaks.size < 2:
            return None
        cycle = np.mean(np.diff(peaks))
        return cycle

    @staticmethod
    def convertToTensor(dataset):
        rp = RecurrencePlot(dimension=1, percentage=10)
        rpArray = rp.fit_transform(dataset)
        rpTensor = [torch.tensor(s).float() for s in rpArray]
        return rpTensor

    @staticmethod
    def collectMeanStd(dataset):
        meanList, stdList = [], []
        for seq in dataset:
            meanList.append(seq.mean())
            stdList.append(seq.std())
        return meanList, stdList

    def train(self, train, valid):
        model = VariationalAutoEncoder(image_channels=1)
        model = model.to(device)

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learningRate)
        criterion = nn.L1Loss(reduction='sum').to(device)

        bestModel = copy.deepcopy(model.state_dict())
        bestLoss = np.inf

        # early stop epoch: 10% of max epoch
        earlyStopThreshold = self.maxEpoch * 0.2
        countWithoutImprovement = 0
        for epoch in range(1, self.maxEpoch + 1):
            model = model.train()

            trainLossList = []
            for seqTrue in train:
                optimizer.zero_grad()

                recon, mu, logVar = model(seqTrue.unsqueeze(0).unsqueeze(0).to(device))
                seqTrue = seqTrue.to(device)
                loss = criterion(recon.squeeze(0).squeeze(0), seqTrue)

                loss.backward()
                optimizer.step()

                trainLossList.append(loss.item())

            validLossList = []
            model = model.eval()
            with torch.no_grad():
                for seqTrue in valid:
                    seqTrue = seqTrue.to(device)
                    recon, mu, logVar = model(seqTrue.unsqueeze(0).unsqueeze(0).to(device))
                    loss = criterion(recon.squeeze(0).squeeze(0), seqTrue)

                    validLossList.append(loss.item())

                    MeanOfTrainLoss = np.mean(trainLossList)
                    MeanOfValidLoss = np.mean(validLossList)

            if MeanOfValidLoss < bestLoss:
                countWithoutImprovement = 0
                bestLoss = MeanOfValidLoss
                bestModel = copy.deepcopy(model.state_dict())
            else:
                countWithoutImprovement += 1

            if epoch > 5 and countWithoutImprovement == earlyStopThreshold:
                print('Early stopping!')
                break

            print(f'Epoch {epoch}: train loss {MeanOfTrainLoss} val loss {MeanOfValidLoss}')

            model.load_state_dict(bestModel)

        # plot result [optional]
        fig, axs = plt.subplots(
            nrows=2,
            ncols=6,
            sharex=True,
            sharey=True,
            figsize=(16, 6)
        )

        for i, data in enumerate(valid[:6]):
            self.plotPrediction(data, model, title='truth', ax=axs[0, i])
            self.plotPrediction(data, model, title='reconstruct', ax=axs[1, i])

        fig.tight_layout()

        return model

    def setThreshold(self, variationalAutoEncoder, train, valid):
        _, trainLosses = self.predict(variationalAutoEncoder, train)
        _, validLosses = self.predict(variationalAutoEncoder, valid)

        # plot loss distribution [optional]
        sns.distplot(trainLosses, bins=50, kde=True)
        sns.distplot(validLosses, bins=50, kde=True)

        self.threshold = np.percentile(validLosses, 95)
        self.statistics.append(self.threshold)

    @staticmethod
    def predict(variationalAutoEncoder, dataset):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction='sum').to(device)
        with torch.no_grad():
            variationalAutoEncoder = variationalAutoEncoder.eval()
            for seqTrue in dataset:
                recon, mu, logVar = variationalAutoEncoder(seqTrue.unsqueeze(0).unsqueeze(0).to(device))
                seqTrue = seqTrue.to(device)
                loss = criterion(recon.squeeze(0).squeeze(0), seqTrue)
                predictions.append(recon.squeeze(0).squeeze(0).cpu().numpy())
                losses.append(loss.item())
        return predictions, losses

    def saveModel(self, variationalAutoEncoder):
        np.save('./model/' + str(self.paramIndex) + '_statistics', self.statistics)
        path = './model/' + str(self.paramIndex) + '_cnn_vae_model.pth'
        torch.save(variationalAutoEncoder, path)

    def loadModel(self):
        self.statistics = np.load('./model/' + str(self.paramIndex) + '_statistics.npy')
        self.threshold = self.statistics[-1]
        variationalAutoEncoder = torch.load('./model/' + str(self.paramIndex) + '_cnn_vae_model.pth')
        variationalAutoEncoder = variationalAutoEncoder.to(device)
        return variationalAutoEncoder

    def evaluate(self, variationalAutoEncoder):
        stable = self.normalData.data.x_data
        unstable = self.unstableData.data.x_data

        minMaxScaler = MinMaxScaler()
        minMaxScaler.fit(stable)
        stableStarted = len(unstable) - self.windowSize
        originWindowSize = self.windowSize

        # wait for finding the cycle
        cycle, waitTime = None, 0
        for i in range(stableStarted):
            cycle = self.findCycle(unstable[: i + self.windowSize])
            if cycle is None:
                continue
            else:
                waitTime = i + 1
                break

        isWindowChanged = False
        for i in range(len(unstable) - self.windowSize - waitTime):
            i += waitTime
            # sliding window
            subSequence = unstable[i: i + self.windowSize]

            # re-sampling (normal vs. unstable)
            originCycle = self.statistics[3]
            if cycle > originCycle and isWindowChanged is False:
                self.windowSize *= (cycle / originCycle)
                isWindowChanged = True
                continue
            reSampledSeq = signal.resample(subSequence, np.int(len(subSequence) * np.float(originCycle / cycle)))
            reSampledSeq = reSampledSeq[:originWindowSize]
            mean, std = reSampledSeq.mean(), reSampledSeq.std()

            # flatten dataset and min-max normalize
            reSampledSeq = minMaxScaler.transform(reSampledSeq)
            reSampledSeq = reSampledSeq.reshape(-1, originWindowSize)
            testDataTensor = self.convertToTensor(reSampledSeq)

            prediction, loss = self.predict(variationalAutoEncoder, testDataTensor)

            if loss < self.threshold:
                print(
                    f'Mean lower bound, Mean upper bound, Std upper bound: '
                    f'{np.around(self.statistics[:3], 3)} mean:{np.around(mean, 2)} std: '
                    f'{np.around(std, 3)}')
                print(f'threshold({np.around(self.threshold, 2)}) vs. loss({np.around(loss, 2)})')
                if self.statistics[0] <= mean.item() <= self.statistics[1] and std.item() <= \
                        self.statistics[2]:
                    self.plotFigure(truth=testDataTensor, pred=prediction[0], loss=loss[0])
                    stableStarted = i
                    break

        self.printResult(self.normalData.data.x_data, unstable, stableStarted)

    @staticmethod
    def plotFigure(truth, pred, loss):
        # plot result [optional]
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=(4, 2)
        )
        ax1 = axs[0]
        ax1.imshow(truth[0].cpu().numpy())
        ax1.set_title(f'truth')
        ax2 = axs[1]
        ax2.imshow(pred)
        ax2.set_title(f'loss:{np.around(loss, 2)}')

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
        return

    def plotPrediction(self, data, model, title, ax):
        if title == 'truth':
            ax.imshow(data.cpu().numpy())
            ax.set_title(f'{title}')
        elif title == 'reconstruct':
            predictions, loss = self.predict(model, [data])
            ax.imshow(predictions[0])
            ax.set_title(f'{title} (loss:{np.around(loss[0], 2)})')
