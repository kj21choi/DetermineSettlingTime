import math
from fastdtw import fastdtw
import DataLoader
import numpy as np

WINDOW_SIZE = 12  # 1 = 5 min
EPOCH = 200
PARAM_INDEX = 8726725


# train & test data
normal_data = DataLoader.NormalDataLoader(PARAM_INDEX, 'train')
train_data = normal_data.data.x_data#[:15000]
unstable_data = DataLoader.UnstableDataLoader(PARAM_INDEX, 'test')
eval_data = unstable_data.data.x_data

# minMaxScaler = MinMaxScaler()
# minMaxScaler.fit(train_data.x_data)
# X_train, X_valid = normal_data.divide_data(train_data, False)  # validate


def dtw_approach(train_d, eval_d, index):
    dist = []
    for i in range(math.floor(len(train_d) / WINDOW_SIZE) - 1):
        distance, path = fastdtw(train_d[i*WINDOW_SIZE: (i+1)*WINDOW_SIZE], eval_d)
        dist.append(distance)
    return np.mean(dist)


def calc_threshold(train_d):
    dist = []
    for i in range(math.floor(len(train_d) / WINDOW_SIZE) - 1):
        for j in range(math.floor(len(train_d) / WINDOW_SIZE) - 1 - (i + 1)):
            j = j + (i + 1)
            distance, path = fastdtw(train_d[i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE],
                                     train_d[j * WINDOW_SIZE: (j + 1) * WINDOW_SIZE])
            dist.append(distance)
        print(f'cur({i}), end({math.floor(len(train_d) / WINDOW_SIZE) - 1}), percent[{np.round((i / (math.floor(len(train_d) / WINDOW_SIZE) - 1) * 100), 3)}%], dist({np.round(np.mean(dist), 3)})')
    # return np.percentile(dist, 95)
    return np.mean(dist)


# Sliding window
stableStarted = 4988
isStable = False
# minDistance = 100000
threshold = calc_threshold(train_data)
# threshold = 1.308454038538461 * 1.1
# threshold = 0.3687970377349491 * 1.1
# threshold = 1.0876805000000012
count = 0
for i in range(len(eval_data) - WINDOW_SIZE):
    dist = dtw_approach(train_data, eval_data[i: i + WINDOW_SIZE], i)
    print(f'i:{i}, dist:{dist}, threshold:{threshold}')
    if dist < threshold:
        stableStarted = i
        break

print("stable time:", unstable_data.data.time_axis['act_time'].get(stableStarted))
print("decision time:", unstable_data.data.time_axis['act_time'].get(stableStarted + WINDOW_SIZE - 1))
