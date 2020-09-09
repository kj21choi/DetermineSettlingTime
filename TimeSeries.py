import pandas as pd
from torch.utils.data import Dataset


class TimeSeries(Dataset):
    def __init__(self, paramIndex, seriesType):
        headers = ['param', 'act_time', 'value', 'state', 'dcqv', 'pm']
        data = pd.read_csv('./Dataset/' + seriesType + '/' + str(paramIndex) + '_' + seriesType + '.csv',
                           delimiter=',', names=headers, parse_dates=['act_time'])
        self.len = data.shape[0]
        self.time_axis = data[["act_time"]]
        self.x_data = data[["value"]]

    def __getitem__(self, index):
        return self.x_data[index], self.time_axis[index]

    def __len__(self):
        return self.len


class NormalTimeSeries(TimeSeries):
    """ Normal data"""


class UnstableTimeSeries(TimeSeries):
    """ Unstable data"""

    def __getitem__(self, index):
        return self.time_axis[index]
