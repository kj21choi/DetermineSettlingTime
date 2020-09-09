from sklearn.preprocessing import MinMaxScaler

import DataLoader
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

PARAM_INDEX = 8622145
normal_data = DataLoader.NormalDataLoader(PARAM_INDEX)
train_data = normal_data.data.x_data

unstable_data = DataLoader.UnstableDataLoader(PARAM_INDEX)
eval_data = unstable_data.data.x_data

# Hyper parameters
WINDOW_SIZE = 12  # 1 hour
P_VALUE_THRESHOLD = 0.05

sns.distplot(train_data, label="train")
sns.distplot(eval_data[:200], label="test")
plt.legend()
plt.show()
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


# t-test + ansari-bradley
def conventional_approach(train_d, eval_d, index):
    statistic, p_value = scipy.stats.ttest_ind(train_d, eval_d, equal_var=True)
    AB, p_value_ab = scipy.stats.ansari(train_d, eval_d)

    # statistic, p_value = scipy.stats.ks_2samp(train_d.values.flatten(), eval_d.values.flatten())
    # train_std = train_d.describe().get("value").get("std")
    # eval_std = eval_d.describe().get("value").get("std")
    train_mean = np.mean(train_d)
    eval_mean = np.mean(eval_d)
    train_std = np.std(train_d)
    eval_std = np.std(eval_d)
    print("train_mean, eval_mean:", train_mean, eval_mean)
    print("train_std, eval_std:", train_std, eval_std)
    # print("train mean, std:", train_d.describe().get("value").get("mean"), train_std)
    # print("eval mean, std:", eval_d.describe().get("value").get("mean"), eval_std)
    print("start: ", index + 1)
    # print("t-result: ", AB)
    print("p_value_t: ", p_value)
    print("p_value_ab: ", p_value_ab)
    print("=="*30)
    # if p_value_ab >= P_VALUE_THRESHOLD:
    # if train_std - eval_std >= 0:
    # if p_value > P_VALUE_THRESHOLD:
    if p_value >= P_VALUE_THRESHOLD and p_value_ab >= P_VALUE_THRESHOLD:
        return True
    else:
        return False


# Sliding window
stableStarted = 0
isStable = False
# scaler = MinMaxScaler()
# normalized_train_data = scaler.fit_transform(train_data)
# normalized_train_data = scipy.stats.zscore(train_data)
for i in range(len(eval_data) - WINDOW_SIZE):
    # 1. conventional approach (t-test(mean) + ansari-bradley(var)): pH결과는 좋지만 current 결과 나쁨(중심치 변동)
    # isStable = conventional_approach(train_data, eval_data[i: i + WINDOW_SIZE], i)
    # 1-A. normalize & (ansari-bradley(var)): current는 잘되지만 pH결과 나쁨
    # normalized_eval_data = scipy.stats.zscore(eval_data[i: i + WINDOW_SIZE])
    # normalized_eval_data = scaler.fit_transform(eval_data[i: i + WINDOW_SIZE])
    # isStable = conventional_approach(normalized_train_data, normalized_eval_data, i)
    isStable = conventional_approach(train_data, eval_data[i: i + WINDOW_SIZE], i)

    # 2. time-series approach(ARIMA)
    if isStable:
        stableStarted = i
        break

    # kld = kl_divergence(train_data, eval_data[i: i + WINDOW_SIZE])

print("stable time:", unstable_data.data.time_axis['act_time'].get(stableStarted))
print("decision time:", unstable_data.data.time_axis['act_time'].get(stableStarted + WINDOW_SIZE - 1))

