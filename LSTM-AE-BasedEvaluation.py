import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import DataLoader
import torch
from torch import nn
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from LSTMmodel import RecurrentAutoEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameter
WINDOW_SIZE = 12  # 1 = 5 min
EPOCH = 100
PARAM_INDEX = 8622145

# train & test data
normal_data = DataLoader.NormalDataLoader(PARAM_INDEX)
normal_data.plot_figure()
plt.close()
train_data = normal_data.data

minMaxScaler = MinMaxScaler()
minMaxScaler.fit(train_data.x_data)
# X_train, X_valid = normal_data.divide_data(train_data, True)  # train
X_train, X_valid = normal_data.divide_data(train_data, False)  # validate


# remove some data to reshape
X_train_remove = X_train.shape[0] % WINDOW_SIZE
X_valid_remove = X_valid.shape[0] % WINDOW_SIZE
X_train = X_train[: -X_train_remove]
X_valid = X_valid[: -X_valid_remove]
print("data shape:", X_train.shape, X_valid.shape)
plt.plot(X_train)
plt.plot(X_valid)
plt.show()

# reshape inputs for LSTM [timesteps, samples]
X_train = X_train.reshape(-1, WINDOW_SIZE)  # 12(window)
X_valid = X_valid.reshape(-1, WINDOW_SIZE)
print("data shape:", X_train.shape, X_valid.shape)


# collect mean, std
def collect_mean_std(dataset):
    mean, std = [], []
    for seq in dataset:
        mean.append(seq.mean())
        std.append(seq.std())
    return mean, std


train_mean, train_std = collect_mean_std(X_train)
valid_mean, valid_std = collect_mean_std(X_valid)
train_mean = train_mean + valid_mean
train_std = train_std + valid_std

# 0: left tail, 1: right tail, 2: right tail(std)
mean_std_threshold = [np.percentile(train_mean, 5), np.percentile(train_mean, 95), np.percentile(train_std, 95)]
np.save('./model/'+ str(PARAM_INDEX)+'_mean_std', mean_std_threshold)


# normalize the data
X_train = minMaxScaler.transform(X_train.reshape(-1, 1))
X_valid = minMaxScaler.transform(X_valid.reshape(-1, 1))
print("data shape:", X_train.shape, X_valid.shape)
plt.plot(X_train)
plt.plot(X_valid)
plt.show()

# reshape inputs for LSTM [timesteps, samples]
X_train = X_train.reshape(-1, WINDOW_SIZE)  # 12(window)
X_valid = X_valid.reshape(-1, WINDOW_SIZE)
print("data shape:", X_train.shape, X_valid.shape)


# create dataset
def create_dataset(df):
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in df]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features


train_dataset, seq_len, n_features = create_dataset(X_train)
val_dataset, _, _ = create_dataset(X_valid)

# model
model = RecurrentAutoEncoder(seq_len, n_features, 128)  # why 128??? example 이 140이어서?
model = model.to(device)

# training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss(reduction='sum').to(device)
history = dict(train=[], val=[])

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 10000.0

# early stop
n_epochs_stop = 20
epochs_no_improve = 0

now = datetime.datetime.now()
print(now)
for epoch in range(1, EPOCH + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
        optimizer.zero_grad()

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
        for seq_true in val_dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
        epochs_no_improve = 0
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    else:
        epochs_no_improve += 1

    if epoch > 5 and epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        break

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

now = datetime.datetime.now()
print(now)

model.load_state_dict(best_model_wts)
model.eval()

# save model
MODEL_PATH = './model/pH_'+str(PARAM_INDEX)+'_model.pth'
torch.save(model, MODEL_PATH)


# load model
model = torch.load('./model/pH_'+str(PARAM_INDEX)+'_model.pth')
model = model.to(device)

train_mean_std_load = np.load('./model/'+str(PARAM_INDEX)+'_mean_std.npy')


# choosing a threshold
def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


_, losses = predict(model, train_dataset)
sns.distplot(losses, bins=50, kde=True)

_, losses = predict(model, val_dataset)
sns.distplot(losses, bins=50, kde=True)

THRESHOLD = np.percentile(losses, 95)

correct = sum(l <= THRESHOLD for l in losses)
print(f'Correct normal predictions: {correct}/{len(val_dataset)}')

# Correct normal predictions: 697/734


def plot_prediction(data, model, title, ax):
    predictions, pred_losses = predict(model, [data])

    ax.plot(data, label='true')
    ax.plot(predictions[0], label='reconstructed')
    ax.set_title(f'{title} (loss:{np.around(pred_losses[0], 2)})')
    ax.legend()


fig, axs = plt.subplots(
    nrows=2,
    ncols=6,
    sharex=True,
    sharey=True,
    figsize=(22, 8)
)

for i, data in enumerate(train_dataset[:6]):
    plot_prediction(data, model, title='Train', ax=axs[0, i])

for i, data in enumerate(val_dataset[:6]):
    plot_prediction(data, model, title='Valid', ax=axs[1, i])

fig.tight_layout()

print(THRESHOLD)


# inference data
def plot_figure(truth, pred, loss):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(truth, label='true')
    plt.plot(pred, label='reconstructed')
    plt.title(f'loss:{np.around(loss, 2)}')
    plt.legend()


unstable_data = DataLoader.UnstableDataLoader(PARAM_INDEX)
eval_data = unstable_data.data.x_data
stableStarted = 4988

for i in range(len(eval_data) - WINDOW_SIZE):
    mean = eval_data[i: i + WINDOW_SIZE].mean()
    std = eval_data[i: i + WINDOW_SIZE].std()
    # normalize
    normalized_eval_data = minMaxScaler.transform(eval_data[i: i + WINDOW_SIZE])
    # normalized_eval_data = normalized_eval_data.reshape(�ٴٰ� �鸰��-1, WINDOW_SIZE)
    normalized_eval_data = normalized_eval_data.reshape(-1, WINDOW_SIZE)
    data, _, _ = create_dataset(normalized_eval_data)
    prediction, loss = predict(model, data)
    # plot_figure(truth=normalized_eval_data[0], pred=prediction[0], loss=loss[0])

    if loss <= THRESHOLD:
        print(f'LML/UML/USL: {np.around(train_mean_std_load, 3)} mean:{np.around(mean, 3)} std: {np.around(std, 4)}')
        print(f'threshold({np.around(THRESHOLD, 2)}) vs. loss({np.around(loss, 2)})')
        if train_mean_std_load[0] <= mean.item() <= train_mean_std_load[1] and std.item() <= train_mean_std_load[2]:

            plot_figure(truth=normalized_eval_data[0], pred=prediction[0], loss=loss[0])
            stableStarted = i
            break
        else:
            continue

print("unstable time:", unstable_data.data.time_axis['act_time'].get(0))
print("stable time:", unstable_data.data.time_axis['act_time'].get(stableStarted))
print("decision time:", unstable_data.data.time_axis['act_time'].get(stableStarted + WINDOW_SIZE - 1))
